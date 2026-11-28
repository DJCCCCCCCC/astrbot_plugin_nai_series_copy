"""Resolve only images attached to this event or its explicit replies."""

import asyncio
import base64
import io
import json
import math
import re
from pathlib import Path

import aiohttp
from astrbot.api.message_components import At
from PIL import Image as PILImage
from PIL import ImageOps

from .models import model_family
from .providers import GenerationError, _validate_image, download_image

LIMIT = 24 * 1024 * 1024
# V5 reference sizes from the published NovelAI web client. Landscape sizes mirror these.
NAI5_PORTRAIT_REFERENCE_SIZES = (
    (256, 4096),
    (256, 4032),
    (256, 3968),
    (256, 3904),
    (256, 3840),
    (256, 3776),
    (256, 3712),
    (256, 3648),
    (256, 3584),
    (256, 3520),
    (256, 3456),
    (256, 3392),
    (256, 3328),
    (320, 3264),
    (320, 3200),
    (320, 3136),
    (320, 3072),
    (320, 3008),
    (320, 2944),
    (320, 2880),
    (320, 2816),
    (320, 2752),
    (384, 2688),
    (384, 2624),
    (384, 2560),
    (384, 2496),
    (384, 2432),
    (384, 2368),
    (448, 2304),
    (448, 2240),
    (448, 2176),
    (448, 2112),
    (512, 2048),
    (512, 1984),
    (512, 1920),
    (512, 1856),
    (576, 1792),
    (576, 1728),
    (576, 1664),
    (640, 1600),
    (640, 1536),
    (704, 1472),
    (704, 1408),
    (768, 1344),
    (768, 1280),
    (832, 1216),
    (896, 1152),
    (960, 1088),
)
NAI5_REFERENCE_SIZES = (
    *NAI5_PORTRAIT_REFERENCE_SIZES,
    (1024, 1024),
    *((height, width) for width, height in reversed(NAI5_PORTRAIT_REFERENCE_SIZES)),
)


def inspect_precise_reference(data, reference_type):
    """Inspect returned PNG metadata without treating it as proof of visual similarity.

    Args:
        data: Generated image bytes.
        reference_type: Requested precise reference type.

    Returns:
        A metadata status: not_applied, parameters_present, or unknown.
    """
    try:
        with PILImage.open(io.BytesIO(data)) as image:
            if image.format != "PNG":
                return "unknown"
            comment = image.info.get("Comment")
        if not isinstance(comment, str) or len(comment) > 1024 * 1024:
            return "unknown"
        metadata = json.loads(comment)
        if not isinstance(metadata, dict):
            return "unknown"
        fields = (
            "director_reference_descriptions",
            "director_reference_strengths",
            "director_reference_information_extracted",
            "director_reference_secondary_strengths",
        )
        if all(field in metadata and metadata[field] is None for field in fields):
            return "not_applied"
        descriptions = metadata.get("director_reference_descriptions")
        if isinstance(descriptions, list):
            for item in descriptions:
                if not isinstance(item, dict):
                    continue
                caption = item.get("caption", item)
                if isinstance(caption, dict) and caption.get("base_caption") == reference_type:
                    return "parameters_present"
    except Exception:
        return "unknown"
    return "unknown"


def _data(value):
    if isinstance(value, dict):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump()
    return vars(value) if hasattr(value, "__dict__") else {}


def _scan(value, depth=0):
    """Keep each image's URL/file alternatives together; never inspect text URLs."""
    images, replies, forwards = [], [], []
    if depth > 6:
        raise ValueError("参考消息嵌套过深")
    if isinstance(value, str):
        for kind, fields in re.findall(r"\[CQ:(image|reply|forward),([^\]]+)\]", value):
            values = {}
            for field in fields.split(","):
                key, sep, item = field.partition("=")
                if sep:
                    values[key] = (
                        item.replace("&#44;", ",")
                        .replace("&#91;", "[")
                        .replace("&#93;", "]")
                        .replace("&amp;", "&")
                    )
            nested = _scan({"type": kind, "data": values}, depth + 1)
            images.extend(nested[0])
            replies.extend(nested[1])
            forwards.extend(nested[2])
    elif isinstance(value, (list, tuple)):
        if len(value) > 100:
            raise ValueError("参考消息过长")
        for item in value:
            nested = _scan(item, depth + 1)
            images.extend(nested[0])
            replies.extend(nested[1])
            forwards.extend(nested[2])
    else:
        item = _data(value)
        kind = str(item.get("type", "")).lower().split(".")[-1]
        body = item.get("data") if isinstance(item.get("data"), dict) else item
        if kind == "image":
            refs = tuple(
                dict.fromkeys(str(body[k]) for k in ("url", "file", "path") if body.get(k))
            )
            if refs:
                images.append(refs)
        elif kind == "reply":
            replies.append((body.get("id"), body.get("chain") or item.get("chain")))
        elif kind == "forward":
            if body.get("id"):
                forwards.append(str(body["id"]))
        else:
            for key in ("message", "messages", "content", "chain"):
                if key in body:
                    nested = _scan(body[key], depth + 1)
                    images.extend(nested[0])
                    replies.extend(nested[1])
                    forwards.extend(nested[2])
                    break
    if len(images) > 16 or len(replies) > 4 or len(forwards) > 4:
        raise ValueError("参考消息中的图片或引用过多")
    return images, replies, forwards


async def _action(event, action, **params):
    bot = getattr(event, "bot", None)
    for client in (bot, getattr(bot, "api", None)):
        call = getattr(client, "call_action", None)
        if not callable(call):
            continue
        try:
            result = await asyncio.wait_for(call(action, **params), 8)
            result = _data(result)
            return result.get("data", result)
        except (Exception, asyncio.TimeoutError):
            continue
    return {}


async def _expand(event, scanned, strict=False):
    images, replies, forwards = scanned
    result = list(images)
    for message_id, embedded in replies:
        nested = _scan(embedded)[0] if embedded else []
        if not nested and message_id is not None:
            payload = await _action(event, "get_msg", message_id=message_id)
            if strict and not payload:
                raise ValueError("引用消息读取失败, 请重新引用或直接附图")
            nested, _, nested_forwards = _scan(payload)
            forwards.extend(nested_forwards)
        result.extend(nested)
    if len(forwards) > 4:
        raise ValueError("参考消息中的转发过多")
    for message_id in dict.fromkeys(forwards):
        payload = await _action(event, "get_forward_msg", message_id=message_id)
        if strict and not payload:
            raise ValueError("合并转发读取失败, 请直接附图")
        result.extend(_scan(payload)[0])
    return result


def qq_avatar_prompt(event, prompt):
    """Bind current QQ mentions to ordered visual roles.

    Args:
        event: The current AstrBot event.
        prompt: The parsed command prompt.

    Returns:
        The prompt with mentions replaced by role labels and the ordered QQ IDs.

    Raises:
        ValueError: Mentions cannot be safely mapped to the command prompt.
    """
    if event.get_platform_name() != "aiocqhttp":
        return prompt, ()
    message = getattr(event, "message_obj", None)
    getter = getattr(event, "get_messages", None)
    chain = getter() if callable(getter) else getattr(message, "message", [])
    self_id = str(event.get_self_id() or "")
    identifiers, tokens = [], {}
    for item in chain:
        if not isinstance(item, At):
            continue
        identifier = str(item.qq)
        if identifier in ("all", self_id):
            continue
        if not re.fullmatch(r"[0-9]{5,12}", identifier, flags=re.ASCII):
            raise ValueError("无法识别被提及的 QQ 用户，未提交生图")
        if identifier not in identifiers:
            identifiers.append(identifier)
        if len(identifiers) > 4:
            raise ValueError("每次最多参考 4 个 QQ 用户头像，未提交生图")
        token = re.sub(r"\s+", " ", f"@{item.name or ''}({identifier})")
        if token not in tokens:
            tokens[token] = [identifiers.index(identifier) + 1, 0]
        tokens[token][1] += 1
    if not identifiers:
        return prompt, ()
    prompt = re.sub(r"\s+", " ", prompt)
    for token, (_, count) in tokens.items():
        if prompt.count(token) != count:
            raise ValueError("无法准确识别 @ 用户与描述的位置，请重新发送消息；未提交生图")
    pattern = re.compile(
        "|".join(re.escape(token) for token in sorted(tokens, key=len, reverse=True))
    )
    return pattern.sub(lambda match: f"角色{tokens[match.group()][0]}", prompt), tuple(identifiers)


async def resolve_qq_avatars(ids):
    """Fetch QQ avatars and normalize them for vision translation.

    Args:
        ids: QQ IDs in the same order as the corresponding role labels.

    Returns:
        The corresponding normalized PNG bytes, in the same order.

    Raises:
        ValueError: An avatar could not be obtained or validated.
    """
    if len(ids) > 4 or any(not re.fullmatch(r"[0-9]{5,12}", str(i), re.ASCII) for i in ids):
        raise ValueError("QQ 头像请求无效，未提交生图")

    async def fetch(identifier):
        raw = await download_image(
            f"https://q1.qlogo.cn/g?b=qq&nk={identifier}&s=640",
            aiohttp.ClientTimeout(total=15),
        )
        return await asyncio.to_thread(normalize_image, raw, max_side=512)

    results = await asyncio.gather(
        *(fetch(identifier) for identifier in ids), return_exceptions=True
    )
    if any(isinstance(result, BaseException) for result in results):
        raise ValueError("QQ 头像获取失败，请稍后重试；未提交生图") from None
    return tuple(results)


async def collect_images(event, *, strict=False, fetch_current=True, reply_only=False):
    message = getattr(event, "message_obj", None)
    getter = getattr(event, "get_messages", None)
    chain = getter() if callable(getter) else getattr(message, "message", [])
    current = _scan(chain)
    raw = _scan(getattr(message, "raw_message", None))
    # Prefer original platform references when another plugin replaced temp paths.
    embedded = {str(key): chain for key, chain in current[1] if chain}
    replies = [(key, chain or embedded.get(str(key))) for key, chain in (raw[1] or current[1])]
    combined = (
        ([], replies, []) if reply_only else (raw[0] or current[0], replies, raw[2] or current[2])
    )
    result = await _expand(event, combined, strict=strict)
    if fetch_current and not result and getattr(message, "message_id", None):
        fetched = await _action(event, "get_msg", message_id=message.message_id)
        result = await _expand(event, _scan(fetched))
    unique, seen = [], set()
    for refs in result:
        if seen.intersection(refs):
            continue
        seen.update(refs)
        unique.append(refs)
    if len(unique) > 16:
        raise ValueError("每次最多选择 16 张候选参考图")
    return unique


def _read_file(path):
    # Paths are obtained from platform image components, never tool arguments.
    path = Path(path)
    if not path.is_file() or path.stat().st_size > LIMIT:
        raise ValueError("参考图文件已失效或超过 24 MB")
    with path.open("rb") as stream:
        data = stream.read(LIMIT + 1)
    if len(data) > LIMIT:
        raise ValueError("参考图超过 24 MB")
    return data


async def _read_reference(event, ref, resolve=True):
    if len(ref) > LIMIT * 4 // 3 + 200:
        raise ValueError("参考图超过 24 MB")
    if ref.startswith(("http://", "https://")):
        return await download_image(ref, aiohttp.ClientTimeout(total=30))
    if ref.startswith("base64://"):
        return base64.b64decode(ref[9:], validate=True)
    if ref.startswith("data:image/") and ";base64," in ref:
        return base64.b64decode(ref.split(";base64,", 1)[1], validate=True)
    path = ref.removeprefix("file://")
    if Path(path).is_absolute() and Path(path).is_file():
        return await asyncio.to_thread(_read_file, path)
    if resolve:
        payload = await _action(event, "get_image", file=ref)
        for key in ("url", "file", "path"):
            value = payload.get(key) if isinstance(payload, dict) else None
            if value and str(value) != ref:
                try:
                    return await _read_reference(event, str(value), False)
                except (ValueError, OSError, GenerationError, aiohttp.ClientError):
                    continue
    raise ValueError("参考图已失效, 请重新发图或引用原图消息")


def vision_reference_size(data: bytes) -> str:
    """Select a standard output size using the ordinary reference's orientation.

    Args:
        data: Normalized reference image bytes.

    Returns:
        The standard landscape, portrait, or square size label.
    """
    with PILImage.open(io.BytesIO(data)) as image:
        width, height = image.size
    if width * 5 > height * 6:
        return "横图"
    if height * 5 > width * 6:
        return "竖图"
    return "方图"


def normalize_image(data, size=None, *, precise=False, reference_model="nai5", max_side=None):
    _validate_image(data)
    with PILImage.open(io.BytesIO(data)) as source:
        image = ImageOps.exif_transpose(source)
        if precise and model_family(reference_model) == "nai4":
            canvases = ((1024, 1536), (1472, 1472), (1536, 1024))
            target = min(
                canvases,
                key=lambda dims: abs(math.log(image.width * dims[1] / (image.height * dims[0]))),
            )
            image = ImageOps.contain(image.convert("RGBA"), target, PILImage.Resampling.LANCZOS)
            canvas = PILImage.new("RGB", target, "black")
            canvas.paste(
                image,
                ((target[0] - image.width) // 2, (target[1] - image.height) // 2),
                image.getchannel("A"),
            )
            image = canvas
        elif precise:
            ratio = math.log(image.width / image.height)
            width, height = min(
                NAI5_REFERENCE_SIZES,
                key=lambda dims: abs(ratio - math.log(dims[0] / dims[1])),
            )
            if source.format == "PNG" and source.size == (width, height) and not source.info:
                return data
            image = ImageOps.fit(
                image.convert("RGBA"), (width, height), PILImage.Resampling.LANCZOS
            )
        else:
            image = image.convert("RGB")
        if max_side:
            image.thumbnail((max_side, max_side), PILImage.Resampling.LANCZOS)
        if not precise and size and image.size != size:
            image = image.resize(size, PILImage.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format="PNG")
        value = output.getvalue()
        if len(value) > LIMIT:
            raise ValueError("参考图转换后超过 24 MB")
        return value


async def resolve_image(
    event, index=None, *, candidates=None, precise=False, reference_model="nai5"
):
    async with asyncio.timeout(60):
        if candidates is None:
            candidates = await collect_images(event)
        for attempt in range(2):
            if not candidates:
                raise ValueError("请在同一条消息附图, 或引用一条图片消息")
            if index is None and len(candidates) != 1:
                raise ValueError(
                    f"找到 {len(candidates)} 张图, 请用 --image 序号选择一张; 同消息图片在前, 引用图片在后"
                )
            selected = 1 if index is None else index
            if (
                isinstance(selected, bool)
                or not str(selected).isdigit()
                or not 1 <= int(selected) <= len(candidates)
            ):
                raise ValueError("参考图序号超出范围, 从 1 开始")
            for ref in candidates[int(selected) - 1]:
                try:
                    data = await _read_reference(event, ref)
                    return await asyncio.to_thread(
                        normalize_image, data, precise=precise, reference_model=reference_model
                    )
                except (ValueError, OSError, GenerationError, aiohttp.ClientError, TimeoutError):
                    continue
            message_id = getattr(getattr(event, "message_obj", None), "message_id", None)
            if attempt or not message_id:
                break
            fresh = await _action(event, "get_msg", message_id=message_id)
            recovered = await _expand(event, _scan(fresh))
            if not recovered or recovered == candidates:
                break
            candidates = recovered
        raise ValueError("参考图读取失败, 请重新发图或引用原图消息")
