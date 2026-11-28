"""QQ avatar binding and safe image loading tests."""

import io
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from astrbot.api.message_components import At, Plain, Reply
from astrbot_plugin_nai_series.references import qq_avatar_prompt, resolve_qq_avatars
from PIL import Image


def event(chain, platform="aiocqhttp"):
    return SimpleNamespace(
        get_platform_name=lambda: platform,
        get_self_id=lambda: "99999",
        get_messages=lambda: chain,
    )


class MentionTests(unittest.IsolatedAsyncioTestCase):
    def test_order_repeats_and_self(self):
        chain = [
            At(qq="99999", name="bot"),
            At(qq="12345", name="甲"),
            At(qq="67890", name="用户  乙"),
            At(qq="12345", name="甲"),
            At(qq="all", name="全体成员"),
        ]
        text = "画 @甲(12345) 跟 @用户 乙(67890) 牵手，让 @甲(12345) 笑"
        result, ids = qq_avatar_prompt(event(chain), text)
        self.assertEqual(result, "画 角色1 跟 角色2 牵手，让 角色1 笑")
        self.assertEqual(ids, ("12345", "67890"))

    def test_fake_text_quoted_mentions_and_other_platform_ignored(self):
        chain = [Plain("@甲(12345)"), Reply(id="1", chain=[At(qq="12345", name="甲")])]
        self.assertEqual(qq_avatar_prompt(event(chain), "@甲(12345)"), ("@甲(12345)", ()))
        self.assertEqual(
            qq_avatar_prompt(event([At(qq="12345", name="甲")], "discord"), "@甲(12345)"),
            ("@甲(12345)", ()),
        )

    def test_ambiguous_missing_invalid_or_excess_mentions_fail(self):
        mention = At(qq="12345", name="甲")
        for prompt in ("画人", "@甲(12345) @甲(12345)"):
            with self.assertRaisesRegex(ValueError, "无法准确识别"):
                qq_avatar_prompt(event([mention]), prompt)
        with self.assertRaisesRegex(ValueError, "无法识别"):
            qq_avatar_prompt(event([At(qq="abc", name="甲")]), "@甲(abc)")
        with self.assertRaisesRegex(ValueError, "最多参考 4"):
            qq_avatar_prompt(event([At(qq=str(10000 + n), name="甲") for n in range(5)]), "")

    async def test_fetch_normalizes_and_preserves_order(self):
        stream = io.BytesIO()
        Image.new("RGB", (1024, 800), "red").save(stream, "PNG")
        with patch(
            "astrbot_plugin_nai_series.references.download_image",
            new=AsyncMock(return_value=stream.getvalue()),
        ) as fetch:
            images = await resolve_qq_avatars(("12345", "67890"))
        self.assertEqual(len(images), 2)
        self.assertIn("nk=12345", fetch.await_args_list[0].args[0])
        self.assertIn("nk=67890", fetch.await_args_list[1].args[0])
        with Image.open(io.BytesIO(images[0])) as image:
            self.assertEqual(image.format, "PNG")
            self.assertEqual(image.size, (512, 400))

    async def test_fetch_fails_closed(self):
        with patch(
            "astrbot_plugin_nai_series.references.download_image",
            new=AsyncMock(side_effect=OSError("sensitive")),
        ):
            with self.assertRaisesRegex(ValueError, "头像获取失败") as error:
                await resolve_qq_avatars(("12345",))
            self.assertNotIn("sensitive", str(error.exception))
        with self.assertRaisesRegex(ValueError, "请求无效"):
            await resolve_qq_avatars(("../../host",))
