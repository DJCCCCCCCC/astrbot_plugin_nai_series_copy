"""Nai series commands and silent, durable background image tools."""

import asyncio
import base64
import hashlib
import io
import json
import math
import re
import uuid
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.message_components import At, Image, Plain, Reply
from astrbot.api.star import Context, Star, StarTools
from astrbot.api.web import error_response, json_response
from astrbot.api.web import request as web_request
from astrbot.core.message.message_event_result import ResultContentType
from astrbot.core.platform.astrbot_message import MessageMember
from astrbot.core.star.filter.command import GreedyStr
from PIL import Image as PILImage

from .background import ACTIVE, LABELS, BackgroundQueue
from .commands import generation_args, resolve_size
from .gallery import render_gallery, save_preview
from .models import (
    GenerationRequest,
    Preset,
    SessionState,
    find_preset,
    model_family,
    normalize_model,
    parse_presets,
    reference_strength_max,
    reference_types,
)
from .providers import GenerationError, ProviderRouter
from .references import collect_images, inspect_precise_reference, normalize_image, resolve_image
from .storage import Store
from .tools import make_tools
from .translator import PromptTranslator, split_translated_prompt

NAI_TOOLS = tuple(
    f"{family}_{mode}_image" for family in ("nai4", "nai5") for mode in ("generate", "edit")
)
TEMPLATE_DIR = Path(__file__).parent / "templates"
TRANSLATION_IMAGE_THRESHOLD = 300
CANVAS_REFERENCE_LIMIT = 8 * 1024 * 1024


class BatchCommandWakeFilter(filter.CustomFilter):
    def filter(self, event, cfg):
        return bool(event.is_at_or_wake_command)


class NaiSeriesPlugin(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        self.store = Store(StarTools.get_data_dir("astrbot_plugin_nai_series"))
        self.router = ProviderRouter(config)
        self.translator = PromptTranslator(config)
        selected = self.store.get("routing_preferences", {})
        if selected.get("translator") in self.translator.models:
            self.translator.next_index = self.translator.models.index(selected["translator"])
        for index, provider in enumerate(self.router.providers):
            if provider.name == selected.get("image"):
                self.router.next_index = index
        self.queue = BackgroundQueue(self.store, config, self._process_job, self._completed)
        context.register_web_api(
            "/astrbot_plugin_nai_series/canvas/options",
            self.canvas_options,
            ["GET"],
            "Get canvas options",
        )
        context.register_web_api(
            "/astrbot_plugin_nai_series/canvas/request-mode",
            self.canvas_request_mode,
            ["POST"],
            "Set the global image request mode",
        )
        context.register_web_api(
            "/astrbot_plugin_nai_series/canvas/generate",
            self.canvas_generate,
            ["POST"],
            "Generate an image",
        )
        context.register_web_api(
            "/astrbot_plugin_nai_series/canvas/task/<task_id>",
            self.canvas_task,
            ["GET"],
            "Get an image task",
        )
        context.register_web_api(
            "/astrbot_plugin_nai_series/library", self.library, ["GET"], "Get library data"
        )
        context.register_web_api(
            "/astrbot_plugin_nai_series/library/presets/save",
            self.save_library_preset,
            ["POST"],
            "Save preset",
        )
        self._tool_names = []
        changed = False
        for preset in config.get("presets", []):
            if isinstance(preset, dict) and not preset.get("id"):
                preset["id"] = uuid.uuid4().hex
                changed = True
        for family in ("nai4", "nai5"):
            key = f"llm_preset_{family}"
            name = str(config.get(key, "")).strip()
            if name and name not in ("none", "无"):
                preset = find_preset(self._load_presets(), name, family)
                if preset and config[key] != preset.id:
                    config[key] = preset.id
                    changed = True
        if changed and callable(getattr(config, "save_config", None)):
            config.save_config()
        self._load_presets()
        if config.get("enable_llm_tool", False):
            try:
                self._llm_preset(self._llm_family())
            except ValueError as exc:
                logger.warning("[NaiSeries] %s", exc)
            tools = make_tools(self)
            self.context.add_llm_tools(*tools)
            self._tool_names = [tool.name for tool in tools]

    async def terminate(self):
        await self.queue.close()
        for name in self._tool_names:
            self.context.provider_manager.llm_tools.remove_func(name)

    async def initialize(self):
        manager = getattr(self.context, "platform_manager", None)
        if manager and getattr(manager, "platform_insts", []):
            self.queue.start()

    @filter.on_astrbot_loaded()
    async def on_loaded(self):
        self.queue.start()

    @property
    def pending(self):
        return sum(job["state"] in ACTIVE for job in self.store.jobs())

    @filter.on_llm_request()
    async def silence_tool_status(self, event: AstrMessageEvent, req):
        if event.get_extra("_nai_completion", False):
            if req.func_tool:
                for name in NAI_TOOLS:
                    req.func_tool.remove_tool(name)
            return
        recent = self.store.jobs(self._key(event), limit=4, origin="tool")
        facts = [
            f"{j['request']['model']}: {j['state']}, 原描述 {j.get('original_prompt', '')[:500]}, "
            f"实际提示词 {j['request']['prompt'][:700]}"
            for j in recent
            if j.get("state") in {"delivered", "generated", "sending", "send_unknown", "failed"}
        ]
        if facts:
            req.system_prompt = (req.system_prompt or "") + (
                "\n你最近为当前用户处理的生图记录(以任务状态为准, 不要声称看到了图像细节):\n"
                + "\n".join(facts)
            )
        if not self.config.get("enable_llm_tool", False):
            return
        req.system_prompt = (
            (req.system_prompt or "")
            + f"\n用户要求 NAI 生图时直接调用 {self._llm_family()}_generate_image, "
            "模型和预设由管理员配置决定, 不自行切换. 不输出开场白或进度; 图片完成后会在原会话自然接话."
        )
        if event.get_extra("_nai_send_guard"):
            return
        original_send = event.send

        async def guarded_send(message):
            if getattr(message, "type", None) == "tool_call":
                content = str(getattr(message, "chain", ""))
                if any(name in content for name in NAI_TOOLS):
                    return
            if event.get_extra("_nai_silent"):
                message.chain = [part for part in message.chain if not isinstance(part, Plain)]
                if not message.chain:
                    return
            return await original_send(message)

        # Guard only this event, leaving other tools and all platform APIs untouched.
        event.send = guarded_send
        event.set_extra("_nai_original_send", original_send)
        event.set_extra("_nai_send_guard", True)
        original_stream = getattr(event, "send_streaming", None)
        if original_stream:

            async def guarded_stream(generator, use_fallback=False):
                # Do not leak streamed preambles before the model chooses its tool.
                buffered = [chain async for chain in generator]

                async def replay():
                    for chain in buffered:
                        if not event.get_extra("_nai_silent"):
                            yield chain

                return await original_stream(replay(), use_fallback)

            event.send_streaming = guarded_stream

    @filter.on_using_llm_tool()
    async def before_tool(self, event: AstrMessageEvent, tool, tool_args):
        if not event.get_extra("_nai_send_guard"):
            return
        pending = event.get_extra("_nai_preamble", [])
        event.set_extra("_nai_preamble", [])
        if tool.name in NAI_TOOLS:
            event.set_extra("_nai_silent", True)
        elif pending:
            await event.get_extra("_nai_original_send")(MessageChain(chain=pending))

    @filter.on_agent_done()
    async def agent_done(self, event: AstrMessageEvent, run_context, resp):
        event.set_extra("_nai_agent_done", True)

    @filter.on_decorating_result()
    async def silence_tool_reply(self, event: AstrMessageEvent):
        if not event.get_extra("_nai_send_guard") and not event.get_extra("_nai_silent"):
            return
        result = event.get_result()
        if not result:
            return
        if event.get_extra("_nai_silent"):
            result.chain = [part for part in result.chain if not isinstance(part, Plain)]
            if not result.chain:
                event.clear_result()
        elif result.result_content_type == ResultContentType.LLM_RESULT:
            pending = event.get_extra("_nai_preamble", [])
            if not event.get_extra("_nai_agent_done"):
                event.set_extra("_nai_preamble", pending + result.chain)
                event.clear_result()
            elif pending:
                result.chain = pending + result.chain
                event.set_extra("_nai_preamble", [])

    def _load_presets(self):
        return parse_presets(self.config.get("presets", []))

    def _key(self, event):
        return (
            "session:"
            + hashlib.sha256(
                f"{event.unified_msg_origin}:{event.get_sender_id()}".encode()
            ).hexdigest()
        )

    def _state(self, event):
        raw = self.store.get(self._key(event), {})
        return SessionState(
            model=raw.get("model", self.config.get("default_model", "nai4")),
            presets=raw.get("presets", {}),
            last_request=raw.get("last_request"),
        )

    def _persist(self, event, state):
        self.store.put(self._key(event), asdict(state))

    def _model(self, value):
        if value in ("nai4", "nai5"):
            model = normalize_model(self.config.get(f"model_{value}", value))
            if model_family(model) != value:
                raise ValueError(f"{value} 配置的模型名不属于该模型族")
            return model
        return normalize_model(value)

    def _preset(self, event, model, explicit=None):
        family = model_family(model)
        state = self._state(event)
        name = (
            explicit
            if explicit is not None
            else state.presets.get(family, self.config.get(f"default_preset_{family}", ""))
        )
        if name in ("", "none", "无"):
            return None
        preset = find_preset(self._load_presets(), str(name), model)
        if preset is None:
            raise ValueError(f"{family} 预设 {name} 已不存在, 请重新选择或 /预设 {family} none")
        return preset

    def _llm_family(self):
        family = str(self.config.get("llm_model", "nai5")).strip()
        if family not in ("nai4", "nai5"):
            raise ValueError("聊天生图模型必须选择 nai4 或 nai5")
        return family

    def _llm_preset(self, family):
        name = str(self.config.get(f"llm_preset_{family}", "")).strip()
        if not name:
            name = str(self.config.get(f"default_preset_{family}", "")).strip()
        if name in ("", "none", "无"):
            return "none"
        presets = self._load_presets()
        preset = find_preset(presets, name, family)
        if preset is None:
            available = ", ".join(p.name for p in presets if p.family == family)[:200] or "无"
            raise ValueError(
                f"{family} 聊天生图预设 {name[:80]} 不存在. 请在插件配置中重新选择. "
                f"可用预设: {available}"
            )
        return preset.id

    @property
    def request_mode(self) -> str:
        """Return the global protocol used for new image jobs."""
        return "native" if self.config.get("request_mode") == "native" else "openai"

    def _check_precise_route(self, mode: str, route: str = ""):
        """Check whether an image request can use precise reference.

        Args:
            mode: The request mode selected at submission.
            route: An optional starting route selected by the user.

        Raises:
            ValueError: The mode or route cannot use precise reference.
        """
        if mode != "native":
            raise ValueError("精准参考需要切换到 NovelAI 原生模式")
        supported = self.router.precise_reference_routes
        if not supported:
            raise ValueError("没有启用精准参考的图片线路")
        if route and route not in supported:
            raise ValueError("该线路未启用精准参考, 可选线路: " + ", ".join(supported))

    def _request(self, model, translated, preset, flags=None):
        flags = flags or {}
        family = model_family(model)
        width, height = resolve_size(
            str(flags.get("size", self.config.get(f"size_{family}", "竖图")))
        )
        artist = preset.artist_prompt if preset else ""
        negative = (
            preset.negative_prompt
            if preset and preset.negative_prompt is not None
            else str(self.config.get(f"negative_prompt_{family}", ""))
        )
        steps = flags.get("steps", self.config.get(f"steps_{family}", 28))
        if isinstance(steps, bool) or not re.fullmatch(r"\d+", str(steps)):
            raise ValueError("步数必须是 1-28 的整数")
        if any(isinstance(flags.get(key), bool) for key in ("strength", "noise")):
            raise ValueError("strength/noise 必须在 0-1 之间")
        return GenerationRequest(
            model=model,
            prompt=", ".join(p for p in (artist.strip(), translated.strip()) if p),
            negative_prompt=negative,
            width=width,
            height=height,
            steps=int(steps),
            scale=float(flags.get("scale", self.config.get(f"scale_{family}", 5))),
            sampler=str(
                flags.get("sampler", self.config.get(f"sampler_{family}", "k_euler_ancestral"))
            ),
            noise_schedule=str(
                flags.get("noise_schedule", self.config.get(f"noise_schedule_{family}", "karras"))
            ),
            cfg_rescale=float(
                flags.get("cfg_rescale", self.config.get(f"cfg_rescale_{family}", 0))
            ),
            quality=False,
            action="img2img" if flags.get("img2img") else "generate",
            reference_id=flags.get("reference_id", ""),
            strength=float(flags.get("strength", 0.6)),
            noise=float(flags.get("noise", 0.1)),
            precise_reference_type=flags.get("ref", ""),
            precise_reference_strength=float(flags.get("ref-strength", 1.0)),
            precise_reference_id=flags.get("precise_reference_id", ""),
            request_mode=self.request_mode,
        )

    async def _reference_flags(self, event, model, prompt, flags, describe_reply=False):
        self._check_access(event)
        if "ref-strength" in flags and "ref" not in flags:
            raise ValueError("使用 --ref-strength 时也需要通过 --ref 指定精准参考类型")
        if "ref" in flags:
            if any(key in flags for key in ("img2img", "strength", "noise")):
                raise ValueError("精准参考不能与 --img2img、--strength 或 --noise 同时使用")
            self._request(
                model,
                prompt,
                self._preset(event, model, flags.get("preset")),
                {**flags, "precise_reference_id": "0" * 64},
            )
            self._check_precise_route(
                self.request_mode, str(flags.get("route", "")).strip().lower()
            )
            try:
                async with asyncio.timeout(30):
                    candidates = await collect_images(event, strict=True, fetch_current=False)
                    data = await resolve_image(
                        event,
                        flags.get("image"),
                        candidates=candidates,
                        precise=True,
                        reference_model=model,
                    )
            except TimeoutError:
                raise ValueError("精准参考图读取超时, 请重新附图或引用") from None
            return {
                **flags,
                "precise_reference_id": self.store.save_reference(self._key(event), data),
            }
        candidates = None
        if describe_reply and not any(flags.get(key) for key in ("img2img", "strength", "noise")):
            try:
                async with asyncio.timeout(30):
                    quoted = await collect_images(
                        event, strict=True, fetch_current=False, reply_only=True
                    )
            except TimeoutError:
                raise ValueError("引用消息读取超时, 请重新引用") from None
            if quoted:
                if flags.get("raw"):
                    raise ValueError("引用图片需要视觉翻译, 不能同时使用 --raw")
                if not self.translator.enabled:
                    raise ValueError("引用图片需要启用支持图片的翻译模型, 未提交生图")
                data = await resolve_image(event, flags.get("image"), candidates=quoted)
                if "size" not in flags:
                    with PILImage.open(io.BytesIO(data)) as image:
                        width, height = image.size
                    size = (
                        "横图"
                        if width * 5 > height * 6
                        else "竖图"
                        if height * 5 > width * 6
                        else "方图"
                    )
                    flags = {**flags, "size": size}
                return {
                    **flags,
                    "vision_reference_id": self.store.save_reference(self._key(event), data),
                }
        if not flags.get("img2img"):
            try:
                async with asyncio.timeout(30):
                    candidates = await collect_images(event, strict=True, fetch_current=False)
            except TimeoutError:
                raise ValueError("参考消息读取超时, 请直接附图") from None
            if candidates:
                flags = {**flags, "img2img": True}
        if not flags.get("img2img"):
            if any(k in flags for k in ("strength", "noise", "image")):
                raise ValueError("图生图参数需要参考图, 请附图或引用图片消息")
            return flags
        # Validate generation settings before downloading any reference.
        draft_flags = {**flags, "reference_id": "0" * 64}
        draft = self._request(
            model, prompt, self._preset(event, model, flags.get("preset")), draft_flags
        )
        try:
            data = await resolve_image(event, flags.get("image"), candidates=candidates)
            data = await asyncio.to_thread(normalize_image, data, (draft.width, draft.height))
        except TimeoutError:
            raise ValueError("参考图读取超时, 请重新发图") from None
        return {**flags, "reference_id": self.store.save_reference(self._key(event), data)}

    def _check_access(self, event):
        if self.config.get("admin_only", False) and not event.is_admin():
            raise ValueError("当前仅允许管理员生图")
        allowed = [str(x) for x in self.config.get("allowed_users", [])]
        if allowed and event.get_sender_id() not in allowed and not event.is_admin():
            raise ValueError("当前用户未获得生图权限")

    def _submit(
        self,
        event,
        model,
        prompt,
        *,
        flags=None,
        snapshot=None,
        save_as=None,
        silent=False,
        count=1,
        completion=None,
        origin="command",
    ):
        self._check_access(event)
        flags = flags or {}
        if not snapshot and (
            (not prompt.strip() and not flags.get("vision_reference_id")) or len(prompt) > 12000
        ):
            raise ValueError("请提供 1-12000 字符的画面描述")
        preset = None if snapshot else self._preset(event, model, flags.get("preset"))
        draft = (
            GenerationRequest(**{**snapshot, "request_mode": self.request_mode})
            if snapshot
            else self._request(model, prompt or "image", preset, flags)
        )
        route = str(flags.get("route", "")).strip().lower()
        if route and route not in [provider.name for provider in self.router.providers]:
            raise ValueError("可用线路: " + ", ".join(p.name for p in self.router.providers))
        if draft.precise_reference_type:
            self._check_precise_route(draft.request_mode, route)
        name_getter = getattr(event, "get_sender_name", None)
        if callable(name_getter):
            nickname = str(name_getter() or "").strip()
            if nickname:
                self.store.put(
                    "user_name:" + f"{event.get_platform_name()}:{event.get_sender_id()}",
                    nickname[:200],
                )
        count = self._count(count)
        batch_id = uuid.uuid4().hex[:12] if count > 1 else ""
        payloads = [
            {
                "owner": self._key(event),
                "usage_user": f"{event.get_platform_name()}:{event.get_sender_id()}",
                "usage_day": datetime.now().astimezone().date().isoformat(),
                "route": route,
                "quota_exempt": event.is_admin(),
                "umo": event.unified_msg_origin,
                "request": asdict(draft),
                "original_prompt": prompt,
                "vision_reference_id": flags.get("vision_reference_id", ""),
                "artist": preset.artist_prompt if preset else "",
                "raw": bool(snapshot or flags.get("raw")),
                "silent": silent,
                "preview": asdict(save_as) if save_as else None,
                "batch_id": batch_id,
                "batch_count": count,
                "completion": completion if origin == "tool" else None,
                "origin": origin,
            }
            for _ in range(count)
        ]
        ids = self.queue.submit_many(payloads)
        return ids[0] if count == 1 else ids

    def _count(self, value):
        try:
            count = int(value)
        except (TypeError, ValueError):
            raise ValueError("生图数量必须是整数") from None
        limit = max(1, min(8, int(self.config.get("batch_max_count", 4))))
        if not 1 <= count <= limit:
            raise ValueError(f"每次可生成 1-{limit} 张")
        return count

    async def _completion_target(self, event):
        manager = getattr(self.context, "conversation_manager", None)
        message = getattr(event, "message_obj", None)
        if manager is None or message is None:
            return None
        try:
            cid = await manager.get_curr_conversation_id(event.unified_msg_origin)
        except Exception:
            return None
        if not cid:
            return None
        return {
            "conversation_id": str(cid),
            "platform_id": event.get_platform_id(),
            "message_type": event.get_message_type().value,
            "self_id": str(message.self_id),
            "session_id": str(message.session_id),
            "sender_id": str(event.get_sender_id()),
            "sender_name": str(event.get_sender_name() or ""),
            "group_id": str(message.group_id or ""),
            "source_message_id": str(message.message_id or ""),
        }

    async def _completed(self, job):
        if job.get("origin") != "tool":
            return
        target = job.get("completion")
        if not target or job["state"] in ACTIVE:
            return
        batch_id = job.get("batch_id")
        jobs = (
            [j for j in self.store.jobs(job["owner"]) if j.get("batch_id") == batch_id]
            if batch_id
            else [job]
        )
        if len(jobs) != job.get("batch_count", 1) or any(j["state"] in ACTIVE for j in jobs):
            return
        manager = getattr(self.context, "conversation_manager", None)
        if manager is None:
            return
        try:
            cid = await manager.get_curr_conversation_id(job["umo"])
            if str(cid or "") != target["conversation_id"]:
                return
            conversation = await manager.get_conversation(job["umo"], cid)
            adapter = self.context.get_platform_inst(target["platform_id"])
            if conversation is None or adapter is None:
                return
            facts = "\n".join(
                f"{j['id']}: 模型 {j['request']['model']}, 状态 {j['state']}, "
                f"用户描述 {j.get('original_prompt', '')[:500]}, "
                f"实际提示词 {j['request']['prompt'][:700]}"
                for j in reversed(jobs)
            )
            chain = [At(qq=target["self_id"])] if target["group_id"] else []
            if target.get("source_message_id"):
                chain.append(
                    Reply(
                        id=target["source_message_id"],
                        sender_id=target["sender_id"],
                        sender_nickname=target["sender_name"],
                        message_str="",
                    )
                )
            abm = await StarTools.create_message(
                type=target["message_type"],
                self_id=target["self_id"],
                session_id=target["session_id"],
                sender=MessageMember(user_id=target["sender_id"], nickname=target["sender_name"]),
                message=chain,
                message_str="",
                group_id=target["group_id"],
            )
            event = adapter.create_event(abm)
            event.is_wake = True
            event.is_at_or_wake_command = True
            event.set_extra("_nai_completion", True)
            event.set_extra("_nai_completion_batch", batch_id or job["id"])
            req = event.request_llm(
                prompt=(
                    "后台生图已结束. 根据以下真实任务记录, 用原有人格自然地对原用户说一句话. "
                    "图已经单独发送; 不再调用生图工具, 不要输出任务编号或机械的状态播报, "
                    "不要臆测提示词以外的画面细节.\n" + facts
                ),
                conversation=conversation,
            )
            if not self.store.claim("completion:" + (batch_id or job["id"])):
                return
            event.set_extra("provider_request", req)
            adapter.commit_event(event)
        except Exception:
            logger.exception("[NaiSeries] completion callback failed for task=%s", job["id"])

    async def _send(self, umo, components):
        matched = await asyncio.wait_for(
            self.context.send_message(umo, MessageChain(chain=components)), timeout=120
        )
        if matched is False:
            raise RuntimeError("platform_unavailable")

    async def library(self):
        """Return dashboard library data without provider credentials."""
        if not web_request.username:
            return error_response("请先登录 Dashboard", status_code=401)
        query = getattr(web_request, "query", {})
        view = query.get("view", "")
        try:
            page = max(0, int(query.get("page", 0)))
        except (TypeError, ValueError):
            return error_response("页码无效", status_code=400)
        today = datetime.now().astimezone().date()
        rankings = {}
        with self.store.connect() as db:
            for key, start in (
                ("day", today.isoformat()),
                ("week", (today - timedelta(days=today.weekday())).isoformat()),
                ("total", "0000-01-01"),
            ):
                records = db.execute(
                    "SELECT user_id, COUNT(*) FROM usage WHERE status='generated' AND day >= ? AND day <= ? GROUP BY user_id ORDER BY COUNT(*) DESC, user_id LIMIT 50",
                    (start, today.isoformat()),
                ).fetchall()
                rankings[key] = [
                    {
                        "user": user,
                        "nickname": self.store.get("user_name:" + user, ""),
                        "count": count,
                    }
                    for user, count in records
                ]
        images = []
        canvas_owner = "dashboard:" + hashlib.sha256(web_request.username.encode()).hexdigest()
        visible_jobs = []
        for job in self.store.jobs() if view in ("", "gallery") else []:
            if job.get("origin") == "canvas" and job.get("owner") != canvas_owner:
                continue
            path = self.store.image_path(job["id"])
            if not path.is_file():
                continue
            visible_jobs.append((job, path))
        for job, path in visible_jobs[page * 30 : (page + 1) * 30]:
            try:
                encoded = base64.b64encode(await asyncio.to_thread(path.read_bytes)).decode()
            except OSError:
                continue
            images.append(
                {
                    "image": encoded,
                    "prompt": job.get("original_prompt", ""),
                    "model": job.get("request", {}).get("model", ""),
                    "created": job.get("created", 0),
                }
            )
        presets = []
        for preset in self._load_presets() if view in ("", "presets") else []:
            path = self.store.preview_path(preset)
            try:
                preview = (
                    base64.b64encode(await asyncio.to_thread(path.read_bytes)).decode()
                    if path.is_file()
                    else ""
                )
            except OSError:
                preview = ""
            presets.append(
                {
                    "id": preset.id,
                    "family": preset.family,
                    "name": preset.name,
                    "model": preset.model,
                    "artist_prompt": preset.artist_prompt,
                    "negative_prompt": preset.negative_prompt,
                    "description": preset.description,
                    "preview_prompt": preset.preview_prompt,
                    "preview_size": preset.preview_size,
                    "image": preview,
                }
            )
        return json_response(
            {
                "rankings": rankings,
                "images": images,
                "presets": presets,
                "has_more": len(visible_jobs) > (page + 1) * 30,
            }
        )

    async def save_library_preset(self):
        """Create or update a preset from the authenticated dashboard."""
        if not web_request.username:
            return error_response("请先登录 Dashboard", status_code=401)
        try:
            data = await web_request.json()
            if not isinstance(data, dict):
                raise ValueError("请求内容必须是对象")
            family = data.get("family")
            if family not in ("nai4", "nai5"):
                raise ValueError("请选择 NAI 4.5 或 NAI 5")
            keys = (
                "name",
                "model",
                "artist_prompt",
                "description",
                "preview_prompt",
                "preview_size",
            )
            if any(not isinstance(data.get(key), str) for key in keys):
                raise ValueError("预设字段格式无效")
            if data.get("negative_prompt") is not None and not isinstance(
                data["negative_prompt"], str
            ):
                raise ValueError("负面提示词格式无效")
            name = data["name"].strip()
            model = normalize_model(data["model"].strip())
            if not name or len(data["name"]) > 100 or name.casefold() in ("none", "无"):
                raise ValueError("预设名称不能为空、保留名称或超过 100 字符")
            if not data["model"].strip() or model_family(model) != family:
                raise ValueError("模型与预设模型族不匹配")
            if (
                any(len(data[key]) > 12000 for key in ("artist_prompt", "preview_prompt"))
                or len(data.get("negative_prompt") or "") > 12000
                or len(data["description"]) > 1000
                or len(data["preview_size"]) > 32
            ):
                raise ValueError("预设文本过长")
            width, height = resolve_size(data["preview_size"].strip())
            GenerationRequest(model, "preview", "", width=width, height=height)
            requested_id = data.get("id", "")
            if not isinstance(requested_id, str):
                raise ValueError("预设 ID 格式无效")
            raw_presets = self.config.get("presets", [])
            if not isinstance(raw_presets, list):
                raise ValueError("当前预设配置不是列表，请先在插件设置中修复")
            presets = self._load_presets()
            existing = (
                next((p for p in presets if p.id == requested_id and p.family == family), None)
                if requested_id
                else None
            )
            if requested_id and existing is None:
                raise ValueError("预设不存在或所属模型族已改变")
            preset_id = existing.id if existing else uuid.uuid4().hex
            for item in presets:
                if (
                    item.family == family
                    and item is not existing
                    and (
                        name.casefold() in (item.name.casefold(), item.id.casefold())
                        or preset_id.casefold() in (item.name.casefold(), item.id.casefold())
                    )
                ):
                    raise ValueError("预设名称或 ID 与现有预设冲突")
            entry = {
                "__template_key": family,
                "id": preset_id,
                "name": name,
                "model": model,
                "artist_prompt": data["artist_prompt"].strip(),
                "negative_prompt": data.get("negative_prompt"),
                "description": data["description"].strip(),
                "preview_prompt": data["preview_prompt"].strip(),
                "preview_size": data["preview_size"].strip(),
            }
            updated = list(raw_presets)
            if existing:
                index = next(
                    index
                    for (index, _), item in zip(
                        ((i, raw) for i, raw in enumerate(raw_presets) if parse_presets([raw])),
                        presets,
                    )
                    if item is existing
                )
                if isinstance(updated[index], dict):
                    entry = {**updated[index], **entry}
                updated[index] = entry
            else:
                updated.append(entry)
            save = getattr(self.config, "save_config", None)
            if not callable(save):
                raise ValueError("当前配置不支持保存预设")
            if existing and existing.name != name:
                with self.store.connect() as db:
                    for key, value in db.execute(
                        "SELECT key, value FROM state WHERE key LIKE 'session:%'"
                    ).fetchall():
                        state = json.loads(value)
                        selection = state.get("presets", {}) if isinstance(state, dict) else {}
                        if (
                            isinstance(selection, dict)
                            and str(selection.get(family, "")).casefold()
                            == existing.name.casefold()
                        ):
                            selection[family] = preset_id
                            db.execute(
                                "UPDATE state SET value=? WHERE key=?",
                                (json.dumps(state, ensure_ascii=False), key),
                            )
            old_values = {
                key: self.config.get(key)
                for key in ("presets", f"default_preset_{family}", f"llm_preset_{family}")
            }
            self.config["presets"] = updated
            if existing:
                for key in (f"default_preset_{family}", f"llm_preset_{family}"):
                    if str(self.config.get(key, "")).casefold() == existing.name.casefold():
                        self.config[key] = preset_id
            try:
                save()
            except Exception:
                for key, value in old_values.items():
                    if value is None:
                        self.config.pop(key, None)
                    else:
                        self.config[key] = value
                raise
            return json_response({"id": preset_id})
        except (ValueError, TypeError) as exc:
            return error_response(str(exc), status_code=400)
        except Exception:
            logger.exception("[NaiSeries] Failed to save preset")
            return error_response("保存预设失败，请查看日志", status_code=500)

    async def canvas_options(self):
        """Return canvas settings without exposing credentials."""
        if not web_request.username:
            return error_response("请先登录 Dashboard", status_code=401)
        presets = self._load_presets()
        families = {}
        for family in ("nai4", "nai5"):
            model = self._model(family)
            families[family] = {
                "presets": [{"id": p.id, "name": p.name} for p in presets if p.family == family],
                "default_preset": str(self.config.get(f"default_preset_{family}", "")),
                "steps": self.config.get(f"steps_{family}", 28),
                "scale": self.config.get(f"scale_{family}", 5),
                "cfg_rescale": self.config.get(f"cfg_rescale_{family}", 0),
                "sampler": self.config.get(f"sampler_{family}", "k_euler_ancestral"),
                "noise_schedule": self.config.get(f"noise_schedule_{family}", "karras"),
                "reference_types": reference_types(model),
                "reference_strength_max": reference_strength_max(model),
            }
        return json_response(
            {
                "families": families,
                "request_mode": self.request_mode,
                "precise_reference_available": (
                    self.request_mode == "native" and bool(self.router.precise_reference_routes)
                ),
                "reference_max_bytes": CANVAS_REFERENCE_LIMIT,
                "translator_enabled": self.translator.enabled,
                "translators": [
                    name for name, url, key, _ in self.translator.connections if url and key
                ],
            }
        )

    async def canvas_request_mode(self):
        """Save the global mode for generation requests from the dashboard and chat."""
        if not web_request.username:
            return error_response("请先登录 Dashboard", status_code=401)
        data = await web_request.json(default={})
        mode = data.get("mode") if isinstance(data, dict) else None
        if mode not in ("openai", "native"):
            return error_response("请选择有效的请求模式", status_code=400)
        if mode != self.request_mode:
            save = getattr(self.config, "save_config", None)
            if not callable(save):
                return error_response("当前配置无法保存请求模式", status_code=500)
            previous = self.config.get("request_mode")
            self.config["request_mode"] = mode
            try:
                save()
            except Exception:
                if previous is None:
                    self.config.pop("request_mode", None)
                else:
                    self.config["request_mode"] = previous
                logger.exception("[NaiSeries] Failed to save image request mode")
                return error_response("保存请求模式失败，请查看日志", status_code=500)
        return json_response(
            {
                "request_mode": mode,
                "precise_reference_available": (
                    mode == "native" and bool(self.router.precise_reference_routes)
                ),
            }
        )

    async def canvas_generate(self):
        """Submit a dashboard image task."""
        username = web_request.username
        if not username:
            return error_response("请先登录 Dashboard", status_code=401)
        data = await web_request.json(default={})
        if not isinstance(data, dict):
            return error_response("请求格式无效", status_code=400)
        mode = self.request_mode
        if data.get("request_mode", mode) != mode:
            return error_response("请求模式已变更，请刷新画布后再提交", status_code=409)
        try:
            prompt = data.get("prompt", "")
            family = data.get("model", "nai4")
            if (
                not isinstance(prompt, str)
                or not prompt.strip()
                or len(prompt) > 12000
                or family not in ("nai4", "nai5")
            ):
                raise ValueError("请输入描述并选择有效模型")
            size = data.get("size", "1024x1024")
            if size not in (
                "1024x1024",
                "768x1280",
                "1280x768",
                "1024x1536",
                "1536x1024",
                "832x1216",
                "1216x832",
            ):
                raise ValueError("请选择有效尺寸")
            model = self._model(family)
            preset_id = data.get("preset", "")
            if not isinstance(preset_id, str):
                raise ValueError("预设无效")
            preset = find_preset(self._load_presets(), preset_id, model) if preset_id else None
            if preset_id and not preset:
                raise ValueError("预设不存在或不属于当前模型")
            translator = data.get("translator", "")
            if not isinstance(translator, str) or (
                translator and translator not in self.translator.models
            ):
                raise ValueError("翻译模型未配置")
            flags = {"size": size}
            for key in ("steps", "scale", "cfg_rescale", "sampler", "noise_schedule"):
                if key in data:
                    flags[key] = data[key]
            for key in ("scale", "cfg_rescale"):
                if key in flags and (
                    isinstance(flags[key], bool) or not math.isfinite(float(flags[key]))
                ):
                    raise ValueError("CFG 参数必须是有效数字")
            for key in ("sampler", "noise_schedule"):
                if key in flags and (
                    not isinstance(flags[key], str)
                    or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", flags[key])
                ):
                    raise ValueError("采样器或噪声调度无效")
            owner = "dashboard:" + hashlib.sha256(username.encode()).hexdigest()
            reference_type = data.get("reference_type", "")
            reference_image = data.get("reference_image", "")
            if not isinstance(reference_type, str) or not isinstance(reference_image, str):
                raise ValueError("参考图参数无效")
            if reference_type or reference_image or "reference_strength" in data:
                self._check_precise_route(mode)
                if not reference_type or not reference_image:
                    raise ValueError("请同时选择参考类型并上传参考图")
                if len(reference_image) > (CANVAS_REFERENCE_LIMIT + 2) // 3 * 4:
                    raise ValueError("参考图不得超过 8 MB")
                strength = data.get("reference_strength", 1.0)
                if isinstance(strength, bool) or not isinstance(strength, (int, float)):
                    raise ValueError("精准参考强度必须是数字")
                flags.update(
                    ref=reference_type,
                    precise_reference_id="0" * 64,
                    **{"ref-strength": strength},
                )
            draft = self._request(model, prompt, preset, flags)
            if reference_type:
                try:
                    raw_image = base64.b64decode(reference_image, validate=True)
                except ValueError:
                    raise ValueError("参考图编码无效") from None
                if len(raw_image) > CANVAS_REFERENCE_LIMIT:
                    raise ValueError("参考图不得超过 8 MB")
                reference = await asyncio.to_thread(
                    normalize_image, raw_image, precise=True, reference_model=model
                )
                if self.request_mode != mode:
                    return error_response("请求模式已变更，请刷新画布后再提交", status_code=409)
                draft = replace(
                    draft, precise_reference_id=self.store.save_reference(owner, reference)
                )
            task_id = self.queue.submit(
                {
                    "owner": owner,
                    "usage_user": owner,
                    "usage_day": datetime.now().astimezone().date().isoformat(),
                    "quota_exempt": True,
                    "umo": "",
                    "request": asdict(draft),
                    "original_prompt": prompt,
                    "artist": preset.artist_prompt if preset else "",
                    "translator": translator,
                    "raw": bool(data.get("raw", False)),
                    "silent": True,
                    "preview": None,
                    "batch_id": "",
                    "batch_count": 1,
                    "completion": None,
                    "origin": "canvas",
                }
            )
            return json_response({"id": task_id})
        except (ValueError, TypeError, GenerationError) as exc:
            return error_response(str(exc), status_code=400)

    async def canvas_task(self, task_id: str):
        """Return the current dashboard task status and result."""
        username = web_request.username
        if not username:
            return error_response("请先登录 Dashboard", status_code=401)
        owner = "dashboard:" + hashlib.sha256(username.encode()).hexdigest()
        job = self.store.job(task_id, owner)
        if not job or job.get("origin") != "canvas":
            return error_response("任务不存在", status_code=404)
        result = {
            "state": job["state"],
            "error": job.get("generation_error", "生成失败，请检查插件日志"),
        }
        if job.get("precise_reference_status"):
            result["precise_reference_status"] = job["precise_reference_status"]
        if job["state"] in ("generated", "delivered"):
            try:
                result["image"] = base64.b64encode(
                    await asyncio.to_thread(self.store.image_path(task_id).read_bytes)
                ).decode()
            except OSError:
                return error_response("图片不可用", status_code=404)
        return json_response(result)

    async def _process_job(self, job, queue):
        try:
            if job["state"] != "generated":
                reference = None
                precise_reference_id = job["request"].get("precise_reference_id")
                reference_id = job["request"].get("reference_id") or precise_reference_id
                if reference_id:
                    try:
                        reference = await asyncio.to_thread(
                            self.store.reference_path(reference_id).read_bytes
                        )
                        if precise_reference_id:
                            reference = await asyncio.to_thread(
                                normalize_image,
                                reference,
                                precise=True,
                                reference_model=job["request"]["model"],
                            )
                    except (OSError, ValueError, GenerationError):
                        raise GenerationError("参考图已失效, 未提交生成") from None
                vision_image = None
                if job.get("vision_reference_id"):
                    try:
                        vision_image = await asyncio.to_thread(
                            self.store.reference_path(job["vision_reference_id"]).read_bytes
                        )
                    except OSError:
                        raise GenerationError("引用图片已失效, 未提交生成") from None
                queue.transition(job, "translating")
                prompt = job["original_prompt"]
                try:
                    if vision_image is not None:
                        translated = await self.translator.translate(prompt, image=vision_image)
                    elif job["raw"]:
                        translated = prompt
                    elif job.get("translator"):
                        translated = await self.translator.translate(
                            prompt, start_model=job["translator"]
                        )
                    else:
                        translated = await self.translator.translate(prompt)
                    if not job["raw"]:
                        positive, negative = split_translated_prompt(translated)
                        if translated != prompt and positive == translated:
                            positive = ", ".join(translated.splitlines())
                except ValueError as exc:
                    # Translator errors contain a local summary, never an upstream body.
                    raise GenerationError(str(exc)) from None
                draft = GenerationRequest(**job["request"])
                request = (
                    draft
                    if job["raw"]
                    else replace(
                        draft,
                        prompt=", ".join(p for p in (job["artist"].strip(), positive.strip()) if p),
                        negative_prompt=(
                            ", ".join(p for p in (draft.negative_prompt.strip(), negative) if p)
                            if negative
                            else draft.negative_prompt
                        ),
                    )
                )
                if (
                    not job["silent"]
                    and translated != prompt
                    and self.config.get("show_translated_prompt", False)
                ):
                    components = [Plain(f"翻译: {translated}")]
                    if self.config.get(
                        "translation_as_image", self.config.get("send_text_as_image", True)
                    ) and len(translated) > max(
                        0,
                        int(
                            self.config.get(
                                "translation_image_threshold", TRANSLATION_IMAGE_THRESHOLD
                            )
                        ),
                    ):
                        try:
                            image_path = await asyncio.wait_for(
                                self.html_render(
                                    (TEMPLATE_DIR / "translation.html").read_text(encoding="utf-8"),
                                    {"translation": translated},
                                    return_url=False,
                                    umo=job["umo"],
                                ),
                                timeout=20,
                            )
                            components = [Image.fromFileSystem(image_path)]
                        except Exception as exc:
                            logger.warning(
                                "[NaiSeries] translation render failed: %s", type(exc).__name__
                            )
                    try:
                        await self._send(job["umo"], components)
                    except Exception as exc:
                        result = getattr(exc, "result", None)
                        retcode = result.get("retcode") if isinstance(result, dict) else None
                        logger.warning(
                            "[NaiSeries] task=%s translation preview delivery failed error=%s retcode=%s",
                            job["id"],
                            type(exc).__name__,
                            retcode if type(retcode) is int else None,
                        )
                        raise GenerationError(
                            "翻译已完成，但译文预览发送失败，未提交生图"
                        ) from None
                queue.transition(job, "running", request=asdict(request))
                outbound = request
                if reference is not None:
                    image_field = (
                        "precise_reference_image" if request.precise_reference_type else "image"
                    )
                    outbound = replace(
                        request, **{image_field: base64.b64encode(reference).decode()}
                    )
                if job.get("route"):
                    images, _route = await self.router.generate(outbound, start_route=job["route"])
                else:
                    images, _route = await self.router.generate(outbound)
                if len(images) != 1:
                    raise GenerationError("接口返回图片数量与单张请求不一致")
                path = self.store.image_path(job["id"])
                temp = path.with_suffix(".tmp")
                temp.write_bytes(images[0].data)
                temp.chmod(0o600)
                temp.replace(path)
                state = self.store.get(job["owner"], {})
                state["last_request"] = asdict(request)
                self.store.put(job["owner"], state)
                if request.precise_reference_type:
                    precise_status = await asyncio.to_thread(
                        inspect_precise_reference, images[0].data, request.precise_reference_type
                    )
                    queue.transition(job, "generated", precise_reference_status=precise_status)
                else:
                    queue.transition(job, "generated")
                if job["preview"]:
                    await asyncio.to_thread(
                        save_preview,
                        images[0].data,
                        self.store.preview_path(Preset(**job["preview"])),
                    )
            if job.get("origin") == "canvas":
                queue.transition(job, "delivered")
                return
            image = await asyncio.to_thread(self.store.image_path(job["id"]).read_bytes)
            queue.transition(job, "sending")
            components = [Image.fromBytes(image)]
            if job["request"].get("precise_reference_type"):
                status = job.get("precise_reference_status", "unknown")
                message = {
                    "not_applied": "精准参考未生效：结果元数据显示参考参数为空。图片已生成，未自动重试。",
                    "parameters_present": "结果元数据包含精准参考参数，实际相似度取决于模型。",
                    "unknown": "无法从结果元数据验证精准参考是否生效。图片已生成，未自动重试。",
                }.get(status, "无法从结果元数据验证精准参考是否生效。图片已生成，未自动重试。")
                components.append(Plain(message))
            await self._send(job["umo"], components)
            queue.transition(job, "delivered")
            logger.info(
                "[NaiSeries] delivered task=%s model=%s", job["id"], job["request"]["model"]
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if isinstance(exc, GenerationError):
                queue.transition(
                    job,
                    job["state"],
                    http_status=exc.status,
                    request_id=exc.request_id,
                    generation_error=exc.message,
                )
                logger.warning(
                    "[NaiSeries] task=%s status=%s request_id=%s",
                    job["id"],
                    exc.status,
                    exc.request_id,
                )
            logger.warning(
                "[NaiSeries] task=%s stage=%s error=%s", job["id"], job["state"], type(exc).__name__
            )
            if not job["silent"]:
                try:
                    await self._send(
                        job["umo"],
                        [
                            Plain(
                                f"任务 {job['id']} 未完成"
                                + (f": {exc.message}" if isinstance(exc, GenerationError) else "")
                                + ", 用 /nai任务 查看状态."
                            )
                        ],
                    )
                except Exception:
                    pass
            raise

    async def _run(self, event, family, text, count_override=None, img2img=False):
        event.should_call_llm(True)
        try:
            prompt, flags = generation_args(text)
            if img2img:
                flags["img2img"] = True
            flagged_count = flags.pop("count", None)
            if count_override is not None and flagged_count is not None:
                raise ValueError("批量指令已指定数量, 不要再写 --count")
            count = self._count(
                count_override
                if count_override is not None
                else flagged_count
                if flagged_count is not None
                else 1
            )
            model = self._model(flags.pop("model", family))
            if family in ("nai4", "nai5") and model_family(model) != family:
                raise ValueError(f"请使用 /{model_family(model)} 调用另一模型族")
            flags = await self._reference_flags(event, model, prompt, flags, describe_reply=True)
            self._submit(
                event,
                model,
                prompt,
                flags=flags,
                count=count,
            )
            yield event.plain_result(f"已提交 {count} 张, 图片完成后自动发送.")
        except (ValueError, GenerationError) as exc:
            yield event.plain_result(f"未完成: {exc}")
        except Exception:
            logger.exception("[NaiSeries] generation or delivery failed")
            yield event.plain_result(
                "生成或图片发送失败, 请检查插件日志; 为避免重复计费没有自动重试"
            )

    @filter.command("nai4")
    async def cmd_nai4(self, event: AstrMessageEvent, prompt: GreedyStr = GreedyStr):
        async for item in self._run(event, "nai4", str(prompt or "")):
            yield item

    @filter.command("nai5")
    async def cmd_nai5(self, event: AstrMessageEvent, prompt: GreedyStr = GreedyStr):
        async for item in self._run(event, "nai5", str(prompt or "")):
            yield item

    @filter.regex(r"^\d+nai[45](?:\s|$)")
    @filter.custom_filter(BatchCommandWakeFilter)
    async def cmd_batch(self, event: AstrMessageEvent):
        match = re.fullmatch(r"(\d+)nai([45])(?:\s+([\s\S]*))?", event.get_message_str().strip())
        if match is None:
            return
        async for item in self._run(event, "nai" + match[2], match[3] or "", match[1]):
            yield item

    @filter.command("nai生图")
    async def cmd_generate(self, event: AstrMessageEvent, prompt: GreedyStr = GreedyStr):
        async for item in self._run(event, self._state(event).model, str(prompt or "")):
            yield item

    @filter.command("nai图生图")
    async def cmd_img2img(self, event: AstrMessageEvent, prompt: GreedyStr = GreedyStr):
        async for item in self._run(
            event, self._state(event).model, str(prompt or ""), img2img=True
        ):
            yield item

    def _choose(self, event, family, name):
        model = self._model(family)
        if name not in ("none", "无"):
            preset = find_preset(self._load_presets(), name, model)
            if not preset:
                raise ValueError("没有此预设, 请使用 /nai预设 查看精确名称")
            name = preset.id
        else:
            name = "none"
        state = self._state(event)
        state.presets[model_family(model)] = name
        state.model = family
        self._persist(event, state)
        return f"已切换 {model_family(model)} 预设: {name}. 仅影响你在当前会话的设置."

    @filter.command("预设")
    async def cmd_preset(self, event: AstrMessageEvent, query: GreedyStr = GreedyStr):
        try:
            family, name = str(query).strip().split(maxsplit=1)
            yield event.plain_result(self._choose(event, family, name))
        except ValueError as exc:
            yield event.plain_result(f"{exc}. 用法: /预设 nai4|nai5 预设名")

    @filter.command("预设nai4")
    async def cmd_preset4(self, event: AstrMessageEvent, name: GreedyStr = GreedyStr):
        try:
            yield event.plain_result(self._choose(event, "nai4", str(name)))
        except ValueError as exc:
            yield event.plain_result(str(exc))

    @filter.command("预设nai5")
    async def cmd_preset5(self, event: AstrMessageEvent, name: GreedyStr = GreedyStr):
        try:
            yield event.plain_result(self._choose(event, "nai5", str(name)))
        except ValueError as exc:
            yield event.plain_result(str(exc))

    @filter.command("nai预设")
    async def cmd_gallery(self, event: AstrMessageEvent, query: GreedyStr = GreedyStr):
        try:
            args = str(query or "").split()
            presets = self._load_presets()
            if args and args[0] in ("nai4", "nai5"):
                family = args.pop(0)
                presets = [p for p in presets if p.family == family]
            pages = max(1, math.ceil(len(presets) / 6))
            page = int(args[0]) if args else 1
            if not 1 <= page <= pages:
                raise ValueError(f"页码必须在 1-{pages}")
            if not presets:
                yield event.plain_result("没有配置预设, 请在插件设置的预设列表添加")
                return
            selected = presets[(page - 1) * 6 : page * 6]
            data = await asyncio.to_thread(
                render_gallery,
                selected,
                self.store,
                page,
                pages,
                self.config.get("gallery_font", ""),
            )
            yield event.chain_result([Image.fromBytes(data)])
            yield event.plain_result("\n".join(f"{p.family}: {p.name}" for p in selected))
        except ValueError as exc:
            yield event.plain_result(str(exc))

    def _query_preset(self, query):
        parts = query.strip().split(maxsplit=1)
        if len(parts) == 2 and parts[0] in ("nai4", "nai5"):
            preset = find_preset(self._load_presets(), parts[1], parts[0])
        else:
            preset = find_preset(self._load_presets(), query)
        if not preset:
            raise ValueError("未找到预设, 请用 /nai预设 查看")
        return preset

    async def _attachment(self, event):
        for component in event.get_messages():
            if isinstance(component, Image):
                return await component.convert_to_file_path()
            if isinstance(component, Reply):
                for quoted in component.chain or []:
                    if isinstance(quoted, Image):
                        return await quoted.convert_to_file_path()
                from astrbot.core.utils.quoted_message import extract_quoted_message_images

                references = await extract_quoted_message_images(event, component)
                if references:
                    return (
                        await Image.fromURL(references[0]).convert_to_file_path()
                        if references[0].startswith("http")
                        else references[0]
                    )
        return None

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("nai预设展示")
    async def cmd_bind_preview(self, event: AstrMessageEvent, query: GreedyStr = GreedyStr):
        try:
            preset = self._query_preset(str(query))
            path = await self._attachment(event)
            if not path:
                yield event.plain_result(
                    "请附图或引用图, 再发送 /nai预设展示 [nai4|nai5] 预设名. 自动生成样图使用 /nai预设试画"
                )
                return
            data = await asyncio.to_thread(Path(path).read_bytes)
            await asyncio.to_thread(save_preview, data, self.store.preview_path(preset))
            yield event.plain_result(
                f"已保存 {preset.family}/{preset.name} 展示图. /nai预设 查看图册."
            )
        except ValueError as exc:
            yield event.plain_result(str(exc))

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("nai预设试画")
    async def cmd_sample(self, event: AstrMessageEvent, query: GreedyStr = GreedyStr):
        event.should_call_llm(True)
        try:
            preset = self._query_preset(str(query))
            task_id = self._submit(
                event,
                self._model(preset.family),
                preset.preview_prompt,
                flags={"preset": preset.id, "size": preset.preview_size},
                save_as=preset,
            )
            yield event.plain_result(f"已提交样图任务 {task_id}.")
        except (ValueError, GenerationError) as exc:
            yield event.plain_result(str(exc))

    @filter.command("nai模型")
    async def cmd_model(self, event: AstrMessageEvent, value: GreedyStr = GreedyStr):
        value = str(value or "").strip()
        state = self._state(event)
        if value:
            state.model = value
            self._persist(event, state)
        yield event.plain_result(
            f"/nai生图 当前模型: {self._model(state.model)}. /nai4 和 /nai5 始终使用各自模型配置."
        )

    @filter.command("nai翻译模型")
    async def cmd_translator_model(self, event: AstrMessageEvent, value: GreedyStr = GreedyStr):
        """Select the next translation model for all users."""
        if not event.is_admin():
            yield event.plain_result("仅管理员可切换翻译模型")
            return
        value = str(value or "").strip()
        if not value:
            yield event.plain_result("翻译模型: " + ", ".join(self.translator.models))
            return
        if value not in self.translator.models:
            yield event.plain_result("模型未配置，可用模型: " + ", ".join(self.translator.models))
            return
        self.translator.next_index = self.translator.models.index(value)
        settings = self.store.get("routing_preferences", {})
        settings["translator"] = value
        self.store.put("routing_preferences", settings)
        yield event.plain_result(f"下一次翻译将从 {value} 开始轮换")

    @filter.command("nai线路")
    async def cmd_route(self, event: AstrMessageEvent, value: GreedyStr = GreedyStr):
        """Select the next image provider for all users."""
        if not event.is_admin():
            yield event.plain_result("仅管理员可切换生图线路")
            return
        value = str(value or "").strip().lower()
        names = [provider.name for provider in self.router.providers]
        if value not in names:
            yield event.plain_result("可用线路: " + (", ".join(names) or "无"))
            return
        self.router.next_index = names.index(value)
        settings = self.store.get("routing_preferences", {})
        settings["image"] = value
        self.store.put("routing_preferences", settings)
        yield event.plain_result(f"下一次生图将从 {value} 开始轮换")

    @filter.command("nai排行")
    async def cmd_ranking(self, event: AstrMessageEvent, period: str = "总"):
        """Show daily, weekly or all-time counts of successfully generated images."""
        periods = {
            "日": "日榜",
            "今日": "日榜",
            "日榜": "日榜",
            "周": "周榜",
            "本周": "周榜",
            "周榜": "周榜",
            "总": "总榜",
            "总榜": "总榜",
        }
        label = periods.get(period.strip())
        if not label:
            yield event.plain_result("用法: /nai排行 日、/nai排行 周 或 /nai排行 总")
            return
        today = datetime.now().astimezone().date()
        query = "SELECT user_id, COUNT(*) AS total FROM usage WHERE status='generated'"
        args = ()
        if label == "日榜":
            query += " AND day = ?"
            args = (today.isoformat(),)
        elif label == "周榜":
            query += " AND day >= ? AND day <= ?"
            args = ((today - timedelta(days=today.weekday())).isoformat(), today.isoformat())
        query += " GROUP BY user_id ORDER BY total DESC, user_id LIMIT 10"
        with self.store.connect() as db:
            rows = db.execute(query, args).fetchall()
        entries = []
        for i, (user, count) in enumerate(rows, 1):
            platform, _, identifier = user.partition(":")
            nickname = self.store.get("user_name:" + user, "")
            avatar = (
                f"https://q1.qlogo.cn/g?b=qq&nk={identifier}&s=100"
                if platform in ("aiocqhttp", "qq") and identifier.isdecimal()
                else ""
            )
            entries.append(
                {
                    "rank": i,
                    "user": user,
                    "id": identifier or user,
                    "nickname": nickname,
                    "count": count,
                    "avatar": avatar,
                }
            )
        lines = [
            f"{row['rank']}. {row['nickname'] + ' (' + row['id'] + ')' if row['nickname'] else row['user']}: {row['count']} 张"
            for row in entries
        ]
        text = f"NAI 生图{label}（成功生成）\n" + ("\n".join(lines) if lines else "暂无记录")
        if not self.config.get("ranking_as_image", self.config.get("send_text_as_image", True)):
            yield event.plain_result(text)
            return
        try:
            image_path = await asyncio.wait_for(
                self.html_render(
                    (TEMPLATE_DIR / "ranking.html").read_text(encoding="utf-8"),
                    {
                        "label": label,
                        "rows": entries,
                        "maximum": max((row["count"] for row in entries), default=1),
                    },
                    return_url=False,
                    umo=event.unified_msg_origin,
                ),
                timeout=20,
            )
        except Exception as exc:
            logger.warning("[NaiSeries] ranking render failed: %s", type(exc).__name__)
            yield event.plain_result(text)
            return
        yield event.image_result(image_path)

    @filter.command("nai状态")
    async def cmd_status(self, event: AstrMessageEvent):
        try:
            rows = []
            for family in ("nai4", "nai5"):
                model = self._model(family)
                preset = self._preset(event, model)
                request = self._request(model, "状态检查", preset)
                rows.append(
                    f"{family}: {model}\n预设: {preset.name if preset else '无'} | {request.size} | {request.steps} 步\n负面词: {len(request.negative_prompt)} 字符"
                )
            yield event.plain_result(
                "\n\n".join(rows)
                + f"\n处理中: {self.pending}\nLLM 工具: {'开' if self._tool_names else '关'}"
                + f"\n聊天生图模型: {self._model(self._llm_family())}"
                + f"\n聊天生图预设: {self._llm_preset(self._llm_family())}"
            )
        except ValueError as exc:
            yield event.plain_result(str(exc))

    @filter.command("nai重绘")
    async def cmd_reroll(self, event: AstrMessageEvent):
        event.should_call_llm(True)
        snapshot = self._state(event).last_request
        if not snapshot:
            yield event.plain_result("当前会话没有历史生图请求")
            return
        try:
            task_id = self._submit(event, snapshot["model"], "", snapshot=snapshot)
            yield event.plain_result(f"已提交重绘任务 {task_id}.")
        except (ValueError, GenerationError) as exc:
            yield event.plain_result(str(exc))

    @filter.command("nai参数")
    async def cmd_parameters(self, event: AstrMessageEvent):
        snapshot = self._state(event).last_request
        if not snapshot:
            yield event.plain_result("当前会话没有历史生图参数")
            return
        text = f"上次实际请求\n模型: {snapshot['model']}\n尺寸: {snapshot['width']}x{snapshot['height']} | 步数: {snapshot['steps']} | CFG: {snapshot['scale']}\n正向: {snapshot['prompt']}\n负面: {snapshot['negative_prompt']}"
        for start in range(0, len(text), 1800):
            yield event.plain_result(text[start : start + 1800])

    @filter.command("nai取消")
    async def cmd_cancel(self, event: AstrMessageEvent, task_id: str = ""):
        count = await self.queue.cancel(self._key(event), task_id.strip())
        if count:
            yield event.plain_result(f"已取消 {count} 个本地任务; 已提交的任务仍可能计费.")
        else:
            yield event.plain_result("当前没有你的待处理任务")

    @filter.command("nai任务")
    async def cmd_tasks(self, event: AstrMessageEvent):
        jobs = self.store.jobs(self._key(event), limit=8)
        rejected = self.store.get("rejected:" + self._key(event), "")
        yield event.plain_result(
            "\n".join(
                f"{j['id']} | {j['request']['model']} | {LABELS.get(j['state'], j['state'])}"
                + (f" | HTTP {j['http_status']}" if j.get("http_status") else "")
                + (f" | 请求编号 {j['request_id']}" if j.get("request_id") else "")
                + (f" | {j['generation_error']}" if j.get("generation_error") else "")
                for j in jobs
            )
            + (f"\n上次未提交: {rejected}" if rejected else "")
            or "当前没有生图任务"
        )

    @filter.command("nai取图")
    async def cmd_retrieve(self, event: AstrMessageEvent, task_id: str):
        job = self.store.job(task_id.strip(), self._key(event))
        if not job or not self.store.image_path(job["id"]).is_file():
            yield event.plain_result("未找到可取回的图片, 请用 /nai任务 查看任务编号.")
            return
        if job["state"] in ACTIVE:
            yield event.plain_result("任务正在处理, 请稍后取图.")
            return
        data = await asyncio.to_thread(self.store.image_path(job["id"]).read_bytes)
        yield event.chain_result([Image.fromBytes(data)])

    @filter.command("nai帮助")
    async def cmd_help(self, event: AstrMessageEvent):
        yield event.plain_result(
            "图生图: 附图或引用图片 + /nai图生图 描述 --model nai5\n"
            "精准参考（需切换到 NovelAI 原生模式）\n"
            "附图或引用图片后发送 /nai5 描述 --ref 角色＋风格\n"
            "参考类型有角色、风格、角色＋风格。NAI 5 还支持服装。\n"
            "/nai4、/nai5、/nai生图: 附图自动图生图; 引用图片先生成提示词再文生图\n"
            "--strength 0.6 --noise 0.1; 多图用 --image 1 选择; --img2img 可强制要求参考图\n"
            "步数支持 1-28; /nai重绘 沿用上次参考图重新生成\n"
            "/nai排行 日|周|总: 查看生图排行榜，默认总榜\n"
            "单次指定生图线路: /nai5 描述 --route primary（也可用 secondary 或 third）\n"
            "Nai系列生图\n/nai4 描述 或 /nai5 描述\n批量: /3nai4 描述 或 /3nai5 描述\n选项放描述后: --count 3 --preset 名称 --raw --size 832x1216 --steps 23 --scale 5 --model 完整模型名\n/预设 nai4|nai5 名称 (none 关闭预设)\n/nai预设 [nai4|nai5] [页码]: 查看图册, 不产生生图费用\n/nai预设展示 [nai4|nai5] 名称 + 附图/引用图: 设置展示图(管理员)\n/nai预设试画 [nai4|nai5] 名称: 生成并保存样图(付费, 管理员)\n/nai模型 完整名 + /nai生图 描述\n/nai状态 /nai重绘 /nai任务\n/nai取消 [任务编号]: 取消自己当前会话的任务\n/nai取图 任务编号: 重新发送已保存图片, 不重新生图\n设置按当前用户和会话分别保存, 两模型预设互不影响."
        )

    async def generate_as_tool(self, event, family, kwargs):
        if event.get_extra("_nai_completion", False):
            return None
        event.set_extra("_nai_silent", True)
        event.clear_result()
        if not self.config.get("enable_llm_tool", False):
            return None
        claim = None
        try:
            family = self._llm_family()
            count = self._count(kwargs.get("count", 1))
            prompt = str(kwargs.get("prompt", ""))
            preset = self._llm_preset(family)
            editing = kwargs.get("_img2img", False)
            signature = hashlib.sha256(
                repr(
                    (
                        family,
                        prompt,
                        count,
                        preset,
                        kwargs.get("strength", 0.6),
                        kwargs.get("noise", 0.1),
                        kwargs.get("image_index"),
                    )
                ).encode()
            ).hexdigest()
            submitted = event.get_extra("_nai_submitted", {})
            if signature in submitted:
                return None
            # Reserve before any await so parallel calls cannot enqueue twice.
            submitted[signature] = "preparing"
            event.set_extra("_nai_submitted", submitted)
            claim = (submitted, signature)
            flags = {"preset": preset}
            if editing:
                flags.update(img2img=True)
            for source, target in (
                ("strength", "strength"),
                ("noise", "noise"),
                ("image_index", "image"),
            ):
                if source in kwargs:
                    flags[target] = kwargs[source]
            flags = await self._reference_flags(event, self._model(family), prompt, flags)
            task_id = self._submit(
                event,
                self._model(family),
                prompt,
                flags=flags,
                silent=True,
                count=count,
                completion=await self._completion_target(event),
                origin="tool",
            )
            submitted[signature] = task_id
            event.set_extra("_nai_submitted", submitted)
            self.store.put("rejected:" + self._key(event), "")
        except (ValueError, GenerationError) as exc:
            event.set_extra("_nai_rejected", str(exc))
            self.store.put("rejected:" + self._key(event), str(exc))
            logger.warning("[NaiSeries] tool rejected: %s", exc)
            if not event.get_extra("_nai_rejection_sent", False):
                event.set_extra("_nai_rejection_sent", True)
                try:
                    await self._send(event.unified_msg_origin, [Plain(f"这次没能开始画: {exc}")])
                except Exception:
                    logger.warning("[NaiSeries] rejection notification failed")
        except Exception:
            logger.warning("[NaiSeries] tool submission failed")
        finally:
            if claim and claim[0].get(claim[1]) == "preparing":
                claim[0].pop(claim[1], None)
        # AstrBot treats None as a direct-result tool and ends the agent loop.
        return None
