import asyncio
import base64
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import test_behavior as behavior
from astrbot_plugin_nai_series.models import GenerationRequest
from astrbot_plugin_nai_series.providers import (
    GeneratedImage,
    OfficialNovelAIProvider,
    OpenAIImagesProvider,
)
from astrbot_plugin_nai_series.references import (
    collect_images,
    inspect_precise_reference,
    resolve_image,
)
from test_behavior import Event, png
from test_core import CoreTests


def image_segment(data=None):
    return {
        "type": "image",
        "data": {"file": "base64://" + base64.b64encode(data or png()).decode()},
    }


def message_event(chain, raw=None):
    event = Event()
    event.message_obj = SimpleNamespace(message=chain, raw_message=raw)
    event.get_messages = lambda: chain
    event.bot = SimpleNamespace(call_action=AsyncMock(return_value={}))
    return event


class ReferenceTests(unittest.IsolatedAsyncioTestCase):
    def test_precise_reference_metadata_is_conservative(self):
        from PIL import Image
        from PIL.PngImagePlugin import PngInfo

        fields = (
            "director_reference_descriptions",
            "director_reference_strengths",
            "director_reference_information_extracted",
            "director_reference_secondary_strengths",
        )
        null_fields = dict.fromkeys(fields)
        present = {
            "director_reference_descriptions": [
                {"caption": {"base_caption": "costume", "char_captions": []}}
            ]
        }
        for metadata, expected in (
            (null_fields, "not_applied"),
            (present, "parameters_present"),
            ({"director_reference_descriptions": None}, "unknown"),
            (
                {"director_reference_descriptions": [{"caption": {"base_caption": "style"}}]},
                "unknown",
            ),
            ("invalid", "unknown"),
        ):
            with self.subTest(metadata=metadata):
                info = PngInfo()
                info.add_text(
                    "Comment", metadata if isinstance(metadata, str) else json.dumps(metadata)
                )
                output = io.BytesIO()
                Image.new("RGB", (4, 4)).save(output, format="PNG", pnginfo=info)
                self.assertEqual(inspect_precise_reference(output.getvalue(), "costume"), expected)
        self.assertEqual(inspect_precise_reference(png(), "costume"), "unknown")

    async def test_stale_component_recovers_current_message(self):
        event = message_event([{"type": "image", "file": "/gone/reference.png"}])
        event.message_obj.message_id = 123
        event.bot.call_action.side_effect = [{}, {"message": [image_segment()]}]
        self.assertTrue(await resolve_image(event))
        event.bot.call_action.assert_awaited_with("get_msg", message_id=123)

    async def test_raw_reply_keeps_embedded_component(self):
        event = message_event(
            [{"type": "reply", "id": "123", "chain": [image_segment()]}],
            {"message": [{"type": "reply", "data": {"id": 123}}]},
        )
        self.assertTrue(await resolve_image(event))
        event.bot.call_action.assert_not_awaited()

    async def test_private_network_image_is_rejected(self):
        event = message_event([{"type": "image", "url": "http://127.0.0.1/secret.png"}])
        with self.assertRaisesRegex(ValueError, "读取失败"):
            await resolve_image(event)

    async def test_current_and_reply_image(self):
        for chain in (
            [image_segment()],
            [{"type": "reply", "id": "123", "chain": [image_segment()]}],
        ):
            event = message_event(chain)
            data = await resolve_image(event)
            from PIL import Image

            with Image.open(io.BytesIO(data)) as image:
                self.assertEqual(image.size, (64, 64))
            event.bot.call_action.assert_not_awaited()

    async def test_precise_reference_buckets_preserve_alpha_without_padding(self):
        from PIL import Image

        for input_size, expected in (
            ((80, 120), (832, 1216)),
            ((100, 100), (1024, 1024)),
            ((160, 100), (1280, 768)),
            ((32, 512), (256, 4096)),
        ):
            with self.subTest(input_size=input_size):
                buffer = io.BytesIO()
                Image.new("RGBA", input_size, (255, 0, 0, 128)).save(buffer, format="PNG")
                event = message_event([image_segment(buffer.getvalue())])
                data = await resolve_image(event, precise=True)
                with Image.open(io.BytesIO(data)) as result:
                    self.assertEqual(result.size, expected)
                    self.assertEqual(result.mode, "RGBA")
                    self.assertEqual(result.getpixel((0, 0)), (255, 0, 0, 128))
                    self.assertEqual(result.getpixel((expected[0] - 1, 0)), (255, 0, 0, 128))

    async def test_precise_reference_respects_exif_orientation(self):
        from PIL import Image

        buffer = io.BytesIO()
        image = Image.new("RGB", (120, 80), "white")
        exif = image.getexif()
        exif[274] = 6
        image.save(buffer, format="JPEG", exif=exif)
        result = await resolve_image(
            message_event([image_segment(buffer.getvalue())]), precise=True
        )
        with Image.open(io.BytesIO(result)) as image:
            self.assertEqual(image.size, (832, 1216))

    async def test_reply_fetched_by_id(self):
        event = message_event([{"type": "reply", "data": {"id": 456}}])
        event.bot.call_action.return_value = {"data": {"message": [image_segment()]}}
        self.assertTrue(await resolve_image(event))
        event.bot.call_action.assert_awaited_once_with("get_msg", message_id=456)

    async def test_raw_payload_recovers_cleaned_temp_file(self):
        event = message_event(
            [{"type": "image", "file": "/gone/test.png"}],
            {"message": [image_segment()]},
        )
        self.assertTrue(await resolve_image(event))
        self.assertEqual(len(await collect_images(event)), 1)

    async def test_media_id_and_forward(self):
        event = message_event([{"type": "forward", "data": {"id": "forward-1"}}])
        event.bot.call_action.side_effect = [
            {"messages": [{"content": [{"type": "image", "data": {"file": "opaque.png"}}]}]},
            {"url": image_segment()["data"]["file"]},
        ]
        self.assertTrue(await resolve_image(event))
        self.assertEqual(event.bot.call_action.call_args.args, ("get_image",))

    async def test_no_avatar_or_url_fallback(self):
        for chain in (
            [],
            [{"type": "at", "data": {"qq": "123"}}],
            [{"type": "text", "data": {"text": "https://example.org/a.png"}}],
        ):
            with self.assertRaisesRegex(ValueError, "附图"):
                await resolve_image(message_event(chain))

    async def test_multiple_requires_explicit_selection(self):
        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (128, 64), "blue").save(buffer, format="PNG")
        event = message_event([image_segment(), image_segment(buffer.getvalue())])
        with self.assertRaisesRegex(ValueError, "2 张图"):
            await resolve_image(event)
        self.assertTrue(await resolve_image(event, 2))
        for index in (0, 3, True, 1.5):
            with self.assertRaises(ValueError):
                await resolve_image(event, index)

    async def test_duplicate_current_reply_and_cq_string(self):
        ref = image_segment()["data"]["file"]
        event = message_event([], {"message": f"[CQ:image,file={ref}]"})
        self.assertTrue(await resolve_image(event))
        event = message_event([image_segment(), {"type": "reply", "chain": [image_segment()]}])
        self.assertEqual(len(await collect_images(event)), 1)

    async def test_invalid_reference_does_not_generate(self):
        with self.assertRaisesRegex(ValueError, "读取失败"):
            await resolve_image(message_event([image_segment(b"not an image")]))

    async def test_real_astrbot_components(self):
        if behavior.NaiSeriesPlugin is None:
            self.skipTest("AstrBot unavailable")
        from astrbot.api.message_components import Image, Reply

        for chain in ([Image.fromBytes(png())], [Reply(id="123", chain=[Image.fromBytes(png())])]):
            self.assertTrue(await resolve_image(message_event(chain)))


class PayloadTests(CoreTests):
    def test_http_errors_keep_only_safe_correlation_fields(self):
        from astrbot_plugin_nai_series.providers import response_error

        response = SimpleNamespace(status=502, headers={})
        body = b'{"error":{"request_id":"abc123","message":"NAI upstream HTTP 500; not retried","secret":"do-not-log"}}'
        error = response_error(response, body)
        self.assertEqual(error.request_id, "abc123")
        self.assertIn("上游 HTTP 500", str(error))
        self.assertNotIn("do-not-log", str(error))
        body = b'{"error":{"request_id":"bad\\nheader","message":"a credential"}}'
        error = response_error(response, body)
        self.assertEqual(error.request_id, "")
        self.assertNotIn("credential", str(error))

    async def test_img2img_both_protocols(self):
        encoded = base64.b64encode(png()).decode()
        for cls in (OpenAIImagesProvider, OfficialNovelAIProvider):
            provider = cls("test", f"http://127.0.0.1:{self.port}", "test")
            request = GenerationRequest(
                "nai-diffusion-5-full",
                "cat",
                "",
                action="img2img",
                image=encoded,
                strength=0.7,
                noise=0.2,
            )
            self.assertTrue(await provider.generate(request))
            body = self.requests[-1][1]
            self.assertEqual(body["action"], "img2img")
            parameters = body.get("parameters", body)
            self.assertEqual(parameters["image"], encoded)
            self.assertEqual(parameters["strength"], 0.7)
            self.assertEqual(parameters["noise"], 0.2)

    def test_validation_and_sampler_whitespace(self):
        for steps in (29, 50, 0, True, 28.1):
            with self.assertRaisesRegex(ValueError, "1-28"):
                GenerationRequest("nai5", "cat", "", steps=steps)
        for kwargs in (
            {"action": "img2img"},
            {"strength": float("nan")},
            {"noise": 2},
            {"reference_id": "../file"},
        ):
            with self.assertRaises(ValueError):
                GenerationRequest("nai5", "cat", "", **kwargs)
        self.assertEqual(
            GenerationRequest("nai5", "cat", "", sampler=" k_dpmpp_2s_ancestral ").sampler,
            "k_dpmpp_2s_ancestral",
        )


@unittest.skipIf(behavior.NaiSeriesPlugin is None, "AstrBot unavailable")
class PluginImg2ImgTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = behavior.PluginTests.asyncSetUp
    asyncTearDown = behavior.PluginTests.asyncTearDown
    finish = behavior.PluginTests.finish

    async def test_parallel_tool_calls_enqueue_once(self):
        from astrbot_plugin_nai_series.tools import make_tools

        self.plugin.config["enable_llm_tool"] = True
        event = message_event([image_segment()])
        tool = make_tools(self.plugin)[1]
        context = SimpleNamespace(context=SimpleNamespace(event=event))
        original = self.plugin._reference_flags
        gate = asyncio.Event()

        async def slow(*args):
            await gate.wait()
            return await original(*args)

        with patch.object(self.plugin, "_reference_flags", side_effect=slow):
            first = asyncio.create_task(tool.call(context, prompt="cat"))
            await asyncio.sleep(0)
            await tool.call(context, prompt="cat")
            gate.set()
            await first
        await self.finish()
        self.plugin.router.generate.assert_awaited_once()
        self.assertEqual(len(self.plugin.store.jobs()), 1)

    async def test_cancelled_tool_releases_submission_claim(self):
        from astrbot_plugin_nai_series.tools import make_tools

        self.plugin.config["enable_llm_tool"] = True
        event = message_event([image_segment()])
        tool = make_tools(self.plugin)[1]
        context = SimpleNamespace(context=SimpleNamespace(event=event))
        with patch.object(self.plugin, "_reference_flags", side_effect=asyncio.CancelledError):
            with self.assertRaises(asyncio.CancelledError):
                await tool.call(context, prompt="cat")
        self.assertEqual(event.get_extra("_nai_submitted"), {})
        self.plugin.router.generate.assert_not_awaited()

    async def test_command_persisted_reference_reroll_and_plain_legacy(self):
        event = message_event([image_segment()])
        result = [v async for v in self.plugin.cmd_nai5(event, "cat --img2img --strength 0.8")]
        self.assertIn("已提交", str(result))
        await self.finish()
        request = self.plugin.router.generate.call_args.args[0]
        self.assertEqual(request.action, "img2img")
        self.assertTrue(request.image)
        self.assertEqual(request.strength, 0.8)
        snapshot = self.plugin._state(event).last_request
        self.assertEqual(snapshot["image"], "")
        self.assertTrue(self.plugin.store.reference_path(snapshot["reference_id"]).is_file())
        event.message_obj.message = []
        event.get_messages = lambda: []
        [v async for v in self.plugin.cmd_reroll(event)]
        await self.finish()
        self.assertEqual(self.plugin.router.generate.call_args.args[0], request)
        [v async for v in self.plugin.cmd_nai5(event, "cat")]
        await self.finish()
        self.assertEqual(self.plugin.router.generate.call_args.args[0].action, "generate")

    async def test_missing_image_bad_steps_do_not_translate_or_charge(self):
        event = message_event([])
        for prompt in ("cat --img2img", "cat --img2img --steps 29", "cat --strength 0.4"):
            result = [v async for v in self.plugin.cmd_nai5(event, prompt)]
            self.assertIn("未完成", str(result))
        self.plugin.router.generate.assert_not_awaited()
        self.assertEqual(self.plugin.store.jobs(), [])

    async def test_commands_auto_edit_attached_image(self):
        for handler in (self.plugin.cmd_nai4, self.plugin.cmd_nai5, self.plugin.cmd_generate):
            event = message_event([image_segment()])
            result = [v async for v in handler(event, "cat --strength 0.45")]
            self.assertIn("已提交", str(result))
            await self.finish()
            request = self.plugin.router.generate.call_args.args[0]
            self.assertEqual(request.action, "img2img")
            self.assertEqual(request.strength, 0.45)
            self.assertTrue(request.image)

    async def test_quoted_image_is_translated_with_user_instruction(self):
        event = message_event([{"type": "reply", "data": {"id": 123}}])
        event.bot.call_action.return_value = {"message": [image_segment()]}
        self.plugin.translator.enabled = True
        self.plugin.translator.translate = AsyncMock(return_value="blue hair, cat")
        result = [v async for v in self.plugin.cmd_nai5(event, "帮我将角色改成蓝发")]
        self.assertIn("已提交", str(result))
        await self.finish()
        request = self.plugin.router.generate.call_args.args[0]
        self.assertEqual(request.action, "generate")
        self.assertEqual(request.prompt, "style5, blue hair, cat")
        self.assertFalse(request.image)
        args, kwargs = self.plugin.translator.translate.await_args
        self.assertEqual(args[0], "帮我将角色改成蓝发")
        self.assertTrue(kwargs["image"])
        event.bot.call_action.assert_awaited_once_with("get_msg", message_id=123)
        event.message_obj.message = []
        event.get_messages = lambda: []
        [v async for v in self.plugin.cmd_reroll(event)]
        await self.finish()
        self.plugin.translator.translate.assert_awaited_once()
        self.assertEqual(
            self.plugin.router.generate.await_args.args[0].prompt, "style5, blue hair, cat"
        )

    async def test_quoted_image_without_text_and_explicit_img2img(self):
        event = message_event([{"type": "reply", "data": {"id": 123}}])
        event.bot.call_action.return_value = {"message": [image_segment()]}
        self.plugin.translator.enabled = True
        self.plugin.translator.translate = AsyncMock(return_value="cat")
        result = [v async for v in self.plugin.cmd_nai5(event, "")]
        self.assertIn("已提交", str(result))
        await self.finish()
        self.assertEqual(self.plugin.translator.translate.await_args.args[0], "")
        self.assertEqual(self.plugin.router.generate.await_args.args[0].action, "generate")
        result = [v async for v in self.plugin.cmd_nai5(event, "cat --img2img")]
        self.assertIn("已提交", str(result))
        await self.finish()
        self.assertEqual(self.plugin.router.generate.await_args.args[0].action, "img2img")

    async def test_quoted_vision_size_tracks_reference_unless_explicit(self):
        from PIL import Image

        self.plugin.translator.enabled = True
        self.plugin.translator.translate = AsyncMock(return_value="cat")
        self.plugin.config["size_nai5"] = "大竖图"
        for dimensions, suffix, expected in (
            ((121, 100), "", (1216, 832)),
            ((100, 121), "", (832, 1216)),
            ((120, 100), "", (1024, 1024)),
            ((100, 120), "", (1024, 1024)),
            ((100, 100), "", (1024, 1024)),
            ((121, 100), " --size 768x1280", (768, 1280)),
        ):
            with self.subTest(dimensions=dimensions, suffix=suffix):
                data = io.BytesIO()
                Image.new("RGB", dimensions, "blue").save(data, format="PNG")
                event = message_event([{"type": "reply", "data": {"id": 123}}])
                event.bot.call_action.return_value = {"message": [image_segment(data.getvalue())]}
                result = [item async for item in self.plugin.cmd_nai5(event, "cat" + suffix)]
                self.assertIn("已提交", str(result))
                await self.finish()
                request = self.plugin.router.generate.await_args.args[0]
                self.assertEqual(request.action, "generate")
                self.assertEqual((request.width, request.height), expected)

        event = message_event([{"type": "reply", "data": {"id": 123}}])
        event.bot.call_action.return_value = {"message": [image_segment()]}
        [item async for item in self.plugin.cmd_nai5(event, "cat --img2img")]
        await self.finish()
        request = self.plugin.router.generate.await_args.args[0]
        self.assertEqual(request.action, "img2img")
        self.assertEqual((request.width, request.height), (1024, 1536))

    async def test_precise_reference_command_and_reroll(self):
        event = message_event([{"type": "reply", "data": {"id": 123}}])
        event.bot.call_action.return_value = {"message": [image_segment()]}
        self.plugin.router.precise_reference_routes = ["primary"]
        self.plugin.translator.translate = AsyncMock(return_value="cat in a coat")
        result = [
            item
            async for item in self.plugin.cmd_nai5(
                event, "穿外套的猫 --ref 服装 --ref-strength 2.5 --size 横图"
            )
        ]
        self.assertIn("已提交", str(result))
        await self.finish()
        request = self.plugin.router.generate.await_args.args[0]
        self.assertEqual(request.action, "generate")
        self.assertEqual(request.size, "1216x832")
        self.assertEqual(request.precise_reference_type, "costume")
        self.assertEqual(request.precise_reference_strength, 2.5)
        self.assertTrue(request.precise_reference_image)
        self.assertFalse(request.image)
        self.plugin.translator.translate.assert_awaited_once_with("穿外套的猫")
        snapshot = self.plugin._state(event).last_request
        self.assertEqual(snapshot["precise_reference_image"], "")
        self.plugin.store.prune_jobs(float("inf"), set())
        self.assertTrue(
            self.plugin.store.reference_path(snapshot["precise_reference_id"]).is_file()
        )
        [item async for item in self.plugin.cmd_reroll(event)]
        await self.finish()
        self.assertEqual(self.plugin.router.generate.await_args.args[0], request)

        from PIL import Image

        legacy = io.BytesIO()
        Image.new("RGB", (1024, 1536), "navy").save(legacy, format="PNG")
        owner = self.plugin._key(event)
        state = self.plugin.store.get(owner)
        state["last_request"]["precise_reference_id"] = self.plugin.store.save_reference(
            owner, legacy.getvalue()
        )
        self.plugin.store.put(owner, state)
        [item async for item in self.plugin.cmd_reroll(event)]
        await self.finish()
        request = self.plugin.router.generate.await_args.args[0]
        with Image.open(io.BytesIO(base64.b64decode(request.precise_reference_image))) as image:
            self.assertEqual(image.size, (832, 1216))
        self.assertEqual(request.size, "1216x832")

    async def test_explicit_scene_and_precise_reference_stay_separate(self):
        from PIL import Image

        buffers = []
        for color in ("red", "blue"):
            output = io.BytesIO()
            Image.new("RGB", (64, 64), color).save(output, format="PNG")
            buffers.append(output.getvalue())
        self.plugin.router.precise_reference_routes = ["primary"]
        self.plugin.translator.enabled = True
        self.plugin.translator.translate = AsyncMock(
            return_value="POSITIVE: two people\nUNDESIRED: blurry"
        )
        for flag, expected in (("角色", "character"), ("服装", "costume")):
            with self.subTest(flag=flag):
                self.plugin.translator.translate.reset_mock()
                self.plugin.router.generate.reset_mock()
                event = message_event([image_segment(buffers[0]), image_segment(buffers[1])])
                result = [
                    item
                    async for item in self.plugin.cmd_nai5(event, f"替换人物 --参考 1 --{flag} 2")
                ]
                self.assertIn("已提交", str(result))
                await self.finish()
                args, kwargs = self.plugin.translator.translate.await_args
                self.assertEqual(args[0], "替换人物")
                self.assertTrue(kwargs["scene_reference"])
                with Image.open(io.BytesIO(kwargs["image"])) as image:
                    self.assertEqual(image.convert("RGB").getpixel((0, 0)), (255, 0, 0))
                outbound = self.plugin.router.generate.await_args.args[0]
                self.assertEqual(outbound.action, "generate")
                self.assertFalse(outbound.image)
                self.assertEqual(outbound.precise_reference_type, expected)
                with Image.open(
                    io.BytesIO(base64.b64decode(outbound.precise_reference_image))
                ) as image:
                    self.assertEqual(image.convert("RGB").getpixel((0, 0)), (0, 0, 255))
                self.assertNotEqual(
                    kwargs["image"], base64.b64decode(outbound.precise_reference_image)
                )

    async def test_explicit_reference_options_work_independently(self):
        self.plugin.router.precise_reference_routes = ["primary"]
        self.plugin.translator.enabled = True
        self.plugin.translator.translate = AsyncMock(return_value="POSITIVE: cat")
        event = message_event([image_segment()])
        result = [item async for item in self.plugin.cmd_nai5(event, "猫 --参考 1")]
        self.assertIn("已提交", str(result))
        await self.finish()
        self.assertTrue(self.plugin.translator.translate.await_args.kwargs["image"])
        self.assertFalse(self.plugin.router.generate.await_args.args[0].precise_reference_type)
        self.plugin.translator.translate.reset_mock()
        result = [item async for item in self.plugin.cmd_nai5(event, "猫 --角色 1")]
        self.assertIn("已提交", str(result))
        await self.finish()
        self.assertEqual(
            self.plugin.router.generate.await_args.args[0].precise_reference_type, "character"
        )
        self.assertNotIn("image", self.plugin.translator.translate.await_args.kwargs)

    async def test_explicit_reference_rejects_conflicts_and_bad_indices(self):
        self.plugin.router.precise_reference_routes = ["primary"]
        self.plugin.translator.enabled = True
        self.plugin.translator.translate = AsyncMock(return_value="cat")
        for flags in (
            "--参考 1 --角色 1",
            "--角色 1 --服装 2",
            "--参考 0",
            "--参考 3",
            "--参考 x",
            "--参考 1 --raw",
            "--参考 1 --ref 风格",
            "--参考 1 --image 1",
            "--参考 1 --img2img",
            "--参考 1 --strength 0.5",
            "--参考 1 --noise 0.2",
            "--角色 1 --ref 风格",
            "--角色 1 --image 1",
            "--角色 1 --img2img",
            "--角色 1 --strength 0.5",
            "--角色 1 --noise 0.2",
        ):
            with self.subTest(flags=flags):
                output = io.BytesIO()
                from PIL import Image

                Image.new("RGB", (64, 64), "blue").save(output, format="PNG")
                event = message_event([image_segment(), image_segment(output.getvalue())])
                result = [item async for item in self.plugin.cmd_nai5(event, "猫 " + flags)]
                self.assertIn("未完成", str(result))
        self.plugin.translator.translate.assert_not_awaited()
        self.plugin.router.generate.assert_not_awaited()
        self.assertEqual(self.plugin.store.jobs(), [])

    async def test_precise_metadata_message_and_successful_quota(self):
        from PIL import Image
        from PIL.PngImagePlugin import PngInfo

        fields = (
            "director_reference_descriptions",
            "director_reference_strengths",
            "director_reference_information_extracted",
            "director_reference_secondary_strengths",
        )
        event = message_event([image_segment()])
        self.plugin.router.precise_reference_routes = ["primary"]
        cases = (
            (dict.fromkeys(fields), "not_applied", "精准参考未生效"),
            (
                {"director_reference_descriptions": [{"caption": {"base_caption": "costume"}}]},
                "parameters_present",
                "",
            ),
            (None, "unknown", "无法从结果元数据验证"),
        )
        for metadata, status, text in cases:
            with self.subTest(status=status):
                output = io.BytesIO()
                info = PngInfo()
                if metadata is not None:
                    info.add_text("Comment", json.dumps(metadata))
                Image.new("RGB", (8, 8)).save(output, format="PNG", pnginfo=info)
                self.plugin.router.generate.return_value = (
                    [GeneratedImage(output.getvalue())],
                    "primary",
                )
                self.plugin.context.send_message.reset_mock()
                result = [item async for item in self.plugin.cmd_nai5(event, "cat --ref costume")]
                self.assertIn("已提交", str(result))
                await self.finish()
                job = self.plugin.store.jobs(owner=self.plugin._key(event), limit=1)[0]
                self.assertEqual(job["state"], "delivered")
                self.assertEqual(job["precise_reference_status"], status)
                chain = self.plugin.context.send_message.await_args.args[1].chain
                self.assertEqual(len(chain), 2 if text else 1)
                if text:
                    self.assertIn(text, chain[1].text)
                with self.plugin.store.connect() as db:
                    self.assertEqual(
                        db.execute(
                            "SELECT status FROM usage WHERE job_id=?", (job["id"],)
                        ).fetchone()[0],
                        "generated",
                    )
        self.assertEqual(self.plugin.router.generate.await_count, 3)

    async def test_qq_mentions_are_separate_ordinary_avatar_references(self):
        from astrbot.api.message_components import At, Plain

        event = message_event(
            [
                At(qq="10001", name="甲"),
                Plain("与"),
                At(qq="10002", name="乙"),
                Plain("打架"),
            ]
        )
        event.get_platform_name = lambda: "aiocqhttp"
        event.get_self_id = lambda: "90001"
        self.plugin.translator.enabled = True
        self.plugin.translator.translate = AsyncMock(
            return_value=(
                "POSITIVE: two characters, fighting\n"
                "UNDESIRED: blurry\n"
                "CHAR2: blue hair, dodging\n"
                "CHAR1: red hair, advancing"
            )
        )
        data = (png(), png())
        with patch(
            "astrbot_plugin_nai_series.main.resolve_qq_avatars", AsyncMock(return_value=data)
        ) as resolve:
            results = [
                item async for item in self.plugin.cmd_nai5(event, "@甲(10001) 与 @乙(10002) 打架")
            ]
            self.assertIn("已提交", str(results))
            resolve.assert_awaited_once_with(("10001", "10002"))
            await self.finish()
            self.plugin.translator.translate.assert_awaited_once_with(
                "角色1 与 角色2 打架", avatars=data
            )
            sent = self.plugin.router.generate.await_args.args[0]
            self.assertEqual(sent.character_prompts, ("red hair, advancing", "blue hair, dodging"))
            self.assertEqual(sent.action, "generate")
            self.assertFalse(sent.precise_reference_type)
            self.assertIn("two characters, fighting", sent.prompt)
            self.assertNotIn("red hair", sent.prompt)
            self.assertIn("blurry", sent.negative_prompt)
            self.plugin.translator.translate.return_value = "POSITIVE: cat\nCHAR1: red hair"
            [item async for item in self.plugin.cmd_nai5(event, "@甲(10001) 与 @乙(10002) 打架")]
            await self.finish()
            self.assertEqual(self.plugin.router.generate.await_count, 1)
            records = self.plugin.store.jobs(owner=self.plugin._key(event), limit=1)
            self.assertEqual(records[0]["state"], "failed")
            self.assertIn("分别描述", records[0]["generation_error"])

    async def test_precise_reference_invalid_requests_never_generate(self):
        self.plugin.router.precise_reference_routes = ["primary"]
        self.plugin.router.providers = [
            SimpleNamespace(name="primary"),
            SimpleNamespace(name="secondary"),
        ]
        event = message_event([image_segment()])
        for handler, text, error in (
            (self.plugin.cmd_nai4, "cat --ref costume", "仅支持 NAI 5"),
            (
                self.plugin.cmd_nai5,
                "cat --ref unknown",
                "精准参考类型有服装、角色、风格、角色＋风格",
            ),
            (self.plugin.cmd_nai5, "cat --ref-strength 2", "指定精准参考类型"),
            (self.plugin.cmd_nai5, "cat --ref costume --img2img", "不能与 --img2img"),
            (self.plugin.cmd_nai5, "cat --ref costume --ref-strength 6", "0–5"),
            (self.plugin.cmd_nai5, "cat --ref style --route secondary", "精准参考"),
            (self.plugin.cmd_nai5, "cat --ref style --生站 secondary", "精准参考"),
        ):
            with self.subTest(text=text):
                result = [item async for item in handler(event, text)]
                self.assertIn(error, str(result))
        result = [
            item async for item in self.plugin.cmd_nai5(message_event([]), "cat --ref costume")
        ]
        self.assertIn("附图", str(result))
        self.assertEqual(self.plugin.store.jobs(), [])
        self.plugin.router.generate.assert_not_awaited()

    async def test_chat_precise_reference_respects_site_capability_and_model(self):
        self.plugin.router.precise_reference_routes = []
        event = message_event([image_segment()])
        results = [item async for item in self.plugin.cmd_nai5(event, "cat --ref costume")]
        self.assertIn("精准参考", str(results))
        self.plugin.router.generate.assert_not_awaited()
        self.assertEqual(self.plugin.store.jobs(), [])

        self.plugin.router.precise_reference_routes = ["primary"]
        help_text = str([item async for item in self.plugin.cmd_help(event)])
        self.assertIn("--ref 角色＋风格", help_text)
        self.assertNotIn("character&style", help_text)
        for handler, name, reference_type in (
            (self.plugin.cmd_nai4, "角色", "character"),
            (self.plugin.cmd_nai4, "风格", "style"),
            (self.plugin.cmd_nai5, "角色＋风格", "character&style"),
            (self.plugin.cmd_nai5, "角色 + 风格", "character&style"),
            (self.plugin.cmd_nai5, "服装", "costume"),
        ):
            with self.subTest(reference_type=reference_type):
                results = [item async for item in handler(event, f"cat --ref {name}")]
                self.assertIn("已提交", str(results))
                await self.finish()
                request = self.plugin.router.generate.await_args.args[0]
                self.assertEqual(request.request_mode, "")
                self.assertEqual(request.precise_reference_type, reference_type)
                self.assertEqual(request.precise_reference_strength, 1.0)

    async def test_quoted_image_translation_errors_do_not_generate(self):
        event = message_event([{"type": "reply", "data": {"id": 123}}])
        event.bot.call_action.return_value = {"message": [image_segment()]}
        result = [v async for v in self.plugin.cmd_nai5(event, "cat")]
        self.assertIn("需要启用", str(result))
        self.plugin.translator.enabled = True
        result = [v async for v in self.plugin.cmd_nai5(event, "cat --raw")]
        self.assertIn("不能同时使用", str(result))
        self.plugin.translator.translate = AsyncMock(side_effect=ValueError("vision error"))
        [v async for v in self.plugin.cmd_nai5(event, "cat")]
        await self.finish()
        self.plugin.router.generate.assert_not_awaited()

    async def test_unknown_reference_does_not_fall_back_to_text(self):
        for chain in (
            [{"type": "reply", "data": {"id": 123}}],
            [{"type": "forward", "data": {"id": "forward-1"}}],
            [image_segment(b"invalid image")],
        ):
            event = message_event(chain)
            result = [v async for v in self.plugin.cmd_nai5(event, "cat")]
            self.assertIn("未完成", str(result))
        self.plugin.router.generate.assert_not_awaited()
        self.assertEqual(self.plugin.store.jobs(), [])

    async def test_text_only_reply_uses_text_generation(self):
        event = message_event([{"type": "reply", "data": {"id": 123}}])
        event.bot.call_action.return_value = {
            "message": [{"type": "text", "data": {"text": "hello"}}]
        }
        [v async for v in self.plugin.cmd_nai5(event, "cat")]
        await self.finish()
        self.assertEqual(self.plugin.router.generate.call_args.args[0].action, "generate")

    async def test_text_generation_does_not_lookup_current_message(self):
        event = message_event([])
        event.message_obj.message_id = 123
        [v async for v in self.plugin.cmd_nai5(event, "cat")]
        await self.finish()
        self.assertEqual(self.plugin.router.generate.call_args.args[0].action, "generate")
        event.bot.call_action.assert_not_awaited()

    async def test_legacy_tool_auto_edits_and_deduplicates_edit_tool(self):
        from astrbot_plugin_nai_series.tools import make_tools

        self.plugin.config["enable_llm_tool"] = True
        event = message_event([image_segment()])
        generate, edit = make_tools(self.plugin)
        self.assertIn("image_index", generate.parameters["properties"])
        context = SimpleNamespace(context=SimpleNamespace(event=event))
        await asyncio.gather(
            generate.call(context, prompt="cat"),
            edit.call(context, prompt="cat", strength=0.6, noise=0.1),
        )
        await self.finish()
        self.plugin.router.generate.assert_awaited_once()
        self.assertEqual(self.plugin.router.generate.call_args.args[0].action, "img2img")

    async def test_auto_mode_requires_selection_for_multiple_images(self):
        event = message_event([image_segment(), image_segment(b"second candidate")])
        result = [v async for v in self.plugin.cmd_nai5(event, "cat")]
        self.assertIn("2 张图", str(result))
        self.plugin.router.generate.assert_not_awaited()
        result = [v async for v in self.plugin.cmd_nai5(event, "cat --image 1")]
        self.assertIn("已提交", str(result))
        await self.finish()
        self.assertEqual(self.plugin.router.generate.call_args.args[0].action, "img2img")

    async def test_edit_tool_preserves_preset_and_deduplicates(self):
        from astrbot_plugin_nai_series.tools import make_tools

        self.plugin.config["enable_llm_tool"] = True
        event = message_event([image_segment()])
        tool = make_tools(self.plugin)[1]
        self.assertNotIn("url", tool.parameters["properties"])
        context = SimpleNamespace(context=SimpleNamespace(event=event))
        for _ in range(2):
            await tool.call(context, prompt="cat", strength=0.5)
        await self.finish()
        request = self.plugin.router.generate.call_args.args[0]
        self.assertEqual(request.action, "img2img")
        self.assertEqual(request.prompt, "style5, cat")
        self.assertEqual(request.negative_prompt, "bad5")
        self.plugin.router.generate.assert_awaited_once()

    async def test_stale_reference_fails_before_provider(self):
        event = message_event([])
        self.plugin._submit(event, "nai5", "cat", flags={"img2img": True, "reference_id": "0" * 64})
        await self.finish()
        self.plugin.router.generate.assert_not_awaited()

    def test_references_are_owner_scoped_and_kept_for_reroll(self):
        store = self.plugin.store
        a = store.save_reference("a", png())
        b = store.save_reference("b", png())
        self.assertNotEqual(a, b)
        store.put("a", {"last_request": {"reference_id": a}})
        store.prune_jobs(float("inf"), set())
        self.assertTrue(store.reference_path(a).exists())
        self.assertFalse(store.reference_path(b).exists())
