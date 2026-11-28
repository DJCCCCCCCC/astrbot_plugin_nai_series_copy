"""Parse only explicit trailing flags; keep NAI weight syntax untouched."""

import re


def generation_args(text):
    flags = {}
    pattern = (
        r"(?:^|\s)--(raw|img2img|preset|model|size|steps|scale|count|"
        r"strength|noise|image|route|ref|ref-strength|参考|角色|服装|生站)(?=\s|$)"
    )
    matches = list(re.finditer(pattern, text))
    if not matches:
        return text.strip(), flags
    prompt = text[: matches[0].start()].strip()
    for index, match in enumerate(matches):
        value = text[
            match.end() : matches[index + 1].start() if index + 1 < len(matches) else len(text)
        ].strip()
        key = match.group(1)
        if key in flags:
            raise ValueError(f"重复参数 --{key}")
        if key in ("raw", "img2img"):
            if value:
                raise ValueError(f"--{key} 不接受参数, 请把描述放在选项前")
            flags[key] = True
        else:
            if not value:
                raise ValueError(f"--{key} 缺少值")
            flags[key] = value.strip('"')
    if "生站" in flags:
        if not flags["生站"].strip():
            raise ValueError("--生站 缺少生图站 ID")
        if "route" in flags:
            raise ValueError("--生站 和 --route 只能选一个")
    return prompt, flags


def resolve_size(value):
    sizes = {
        "竖图": (832, 1216),
        "横图": (1216, 832),
        "方图": (1024, 1024),
        "小方图": (640, 640),
        "小竖图": (512, 768),
        "长竖图": (768, 1280),
        "长横图": (1280, 768),
        "大竖图": (1024, 1536),
        "大横图": (1536, 1024),
    }
    if value in sizes:
        return sizes[value]
    match = re.fullmatch(r"(\d+)[xX×](\d+)", value.strip())
    if not match:
        raise ValueError("尺寸请用 竖图/横图/方图 或 832x1216")
    return int(match[1]), int(match[2])
