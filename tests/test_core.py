import base64
import io
import json
import sys
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from aiohttp import web
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from astrbot_plugin_nai_series.models import (
    GenerationRequest,
    find_preset,
    model_family,
    normalize_model,
    parse_presets,
)
from astrbot_plugin_nai_series.providers import (
    GenerationError,
    OfficialNovelAIProvider,
    OpenAIImagesProvider,
    ProviderRouter,
)

_buffer = io.BytesIO()
Image.new("RGB", (64, 64), "orange").save(_buffer, format="PNG")
PNG = _buffer.getvalue()


class CoreTests(unittest.IsolatedAsyncioTestCase):
    def test_official_json_and_final_sse(self):
        provider = OfficialNovelAIProvider("native", "https://image.novelai.net", "test")
        encoded = base64.b64encode(PNG).decode()
        native = json.dumps({"images": [{"image": encoded, "index": 0, "seed": 12}]}).encode()
        self.assertEqual(provider._parse_json_image(native)[0].data, PNG)
        self.assertEqual(
            provider._parse_json_image(json.dumps({"image": encoded}).encode())[0].data, PNG
        )
        stream = (
            "data: "
            + json.dumps({"image": encoded, "final": False})
            + "\n\ndata: "
            + json.dumps({"image": encoded, "final": True})
            + "\n\ndata: [DONE]\n"
        )
        self.assertEqual(len(provider._parse_sse_images(stream.encode())), 1)

    async def asyncSetUp(self):
        self.requests = []
        self.native_auth = []
        self.native_status = 200

        async def images(request):
            self.requests.append((request.path, await request.json()))
            return web.json_response({"data": [{"b64_json": base64.b64encode(PNG).decode()}]})

        async def official(request):
            self.requests.append((request.path, await request.json()))
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as archive:
                archive.writestr("image_0.png", PNG)
            return web.Response(body=buffer.getvalue(), status=201, content_type="application/zip")

        async def native_relay(request):
            self.requests.append((request.path, await request.json()))
            self.native_auth.append(request.headers.get("Authorization"))
            if self.native_status == 201:
                return web.Response(body=PNG, status=201, content_type="image/png")
            return web.json_response(
                {"images": [{"image": base64.b64encode(PNG).decode()}]},
                status=self.native_status,
            )

        app = web.Application()
        app.router.add_post("/v1/images/generations", images)
        app.router.add_post("/ai/generate-image", official)
        app.router.add_post("/relay/v1/ai/generate-image", native_relay)
        app.router.add_post("/relay/v1/images/generations", images)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]

    async def asyncTearDown(self):
        await self.runner.cleanup()

    def test_aliases_and_model_specific_presets(self):
        self.assertEqual(model_family("nai4"), "nai4")
        self.assertEqual(model_family("nai-diffusion-4-5-curated"), "nai4")
        self.assertEqual(model_family("nai-diffusion-5-full"), "nai5")
        self.assertEqual(normalize_model("nai5"), "nai-diffusion-5-full")
        self.assertEqual(normalize_model("nai4"), "nai-diffusion-4-5-full")
        presets = parse_presets(
            json.dumps(
                [
                    {
                        "name": "4风格",
                        "model": "nai4",
                        "artist_prompt": "style4",
                        "negative_prompt": "bad4",
                    },
                    {
                        "name": "5风格",
                        "model": "nai5",
                        "artist_prompt": "style5",
                        "negative_prompt": "bad5",
                    },
                ],
                ensure_ascii=False,
            )
        )
        self.assertEqual(find_preset(presets, "4风格", "nai4").negative_prompt, "bad4")
        self.assertIsNone(find_preset(presets, "5风格", "nai4"))

    async def test_openai_images_payload_does_not_send_seed(self):
        provider = OpenAIImagesProvider("test", f"http://127.0.0.1:{self.port}/v1", "secret")
        request = GenerationRequest("nai-diffusion-5-full", "style5, apple", "bad5", steps=28)
        images = await provider.generate(request)
        self.assertEqual(images[0].data, PNG)
        path, payload = self.requests[0]
        self.assertEqual(path, "/v1/images/generations")
        self.assertEqual(payload["model"], "nai-diffusion-5-full")
        self.assertNotIn("seed", payload)
        self.assertEqual(payload["negative_prompt"], "bad5")

    async def test_official_zip_payload(self):
        provider = OfficialNovelAIProvider("official", f"http://127.0.0.1:{self.port}", "secret")
        images = await provider.generate(
            GenerationRequest("nai-diffusion-4-5-full", "apple", "bad")
        )
        self.assertEqual(images[0].data, PNG)
        self.assertEqual(self.requests[0][0], "/ai/generate-image")
        self.assertEqual(self.requests[0][1]["parameters"]["negative_prompt"], "bad")
        self.assertEqual(self.requests[0][1]["model"], "nai-diffusion-4-5-full")

    async def test_precise_reference_payloads_and_relay_opt_in(self):
        image = base64.b64encode(PNG).decode()
        request = GenerationRequest(
            "nai-diffusion-5-full",
            "cat",
            "bad",
            precise_reference_type="character&style",
            precise_reference_strength=3.5,
            precise_reference_image=image,
        )
        expected = {
            "director_reference_descriptions": [
                {
                    "caption": {"base_caption": "character&style", "char_captions": []},
                    "legacy_uc": False,
                }
            ],
            "director_reference_images": [image],
            "director_reference_strength_values": [3.5],
            "director_reference_secondary_strength_values": [0.0],
            "director_reference_information_extracted": [1.0],
        }
        official = OfficialNovelAIProvider("official", f"http://127.0.0.1:{self.port}", "secret")
        await official.generate(request)
        payload = self.requests[-1][1]
        self.assertEqual(payload["action"], "generate")
        for key, value in expected.items():
            self.assertEqual(payload["parameters"][key], value)
        self.assertFalse(any("fidelity" in key.lower() for key in payload["parameters"]))
        relay = OpenAIImagesProvider("relay", f"http://127.0.0.1:{self.port}", "secret")
        count = len(self.requests)
        with self.assertRaisesRegex(GenerationError, "未启用"):
            await relay.generate(request)
        self.assertEqual(len(self.requests), count)

    async def test_precise_routing_filters_unsupported_and_preserves_rotation(self):
        request = GenerationRequest(
            "nai-diffusion-5-full",
            "cat",
            "",
            precise_reference_type="style",
            precise_reference_image=base64.b64encode(PNG).decode(),
        )
        ordinary = GenerationRequest("nai-diffusion-5-full", "cat", "")
        unsupported = SimpleNamespace(
            name="primary", supports_precise_reference=False, generate=AsyncMock(return_value=["a"])
        )
        first = SimpleNamespace(
            name="secondary",
            supports_precise_reference=True,
            generate=AsyncMock(return_value=["b"]),
        )
        second = SimpleNamespace(
            name="third", supports_precise_reference=True, generate=AsyncMock(return_value=["c"])
        )
        router = ProviderRouter({})
        router.providers = [unsupported, first, second]
        self.assertEqual(router.precise_reference_routes, ["secondary", "third"])
        self.assertEqual((await router.generate(request))[1], "secondary")
        self.assertEqual((await router.generate(request))[1], "third")
        self.assertEqual((await router.generate(ordinary))[1], "primary")
        with self.assertRaisesRegex(GenerationError, "不支持精准参考"):
            await router.generate(request, start_route="primary")
        self.assertEqual(unsupported.generate.await_count, 1)
        first.generate.side_effect = GenerationError("unavailable", status=401)
        self.assertEqual((await router.generate(request, start_route="secondary"))[1], "third")
        second.generate.reset_mock()
        first.generate.side_effect = GenerationError("uncertain timeout")
        with self.assertRaises(GenerationError):
            await router.generate(request, start_route="secondary")
        second.generate.assert_not_awaited()

    async def test_native_relay_uses_original_nai_payload_only_for_precise_reference(self):
        encoded = base64.b64encode(PNG).decode()
        request = GenerationRequest(
            "nai-diffusion-5-full",
            "cat in a maid outfit",
            "blurry",
            precise_reference_type="costume",
            precise_reference_image=encoded,
        )
        base = f"http://127.0.0.1:{self.port}/relay"
        for url in (base, base + "/v1/"):
            with self.subTest(url=url):
                router = ProviderRouter(
                    {
                        "primary_base_url": url,
                        "primary_api_key": "mock-key",
                        "primary_precise_reference": True,
                        "primary_precise_reference_protocol": "openai",
                        "use_stream": True,
                    }
                )
                images, route = await router.generate(request)
                self.assertEqual((images[0].data, route), (PNG, "primary"))
                path, payload = self.requests[-1]
                self.assertEqual(path, "/relay/v1/ai/generate-image")
                self.assertEqual(self.native_auth[-1], "Bearer mock-key")
                self.assertEqual(set(payload), {"input", "model", "action", "parameters"})
                self.assertEqual(payload["action"], "generate")
                self.assertEqual(payload["model"], request.model)
                self.assertEqual(payload["input"], request.prompt)
                params = payload["parameters"]
                self.assertEqual(params["negative_prompt"], request.negative_prompt)
                self.assertEqual(params["params_version"], 4)
                self.assertEqual(params["image_format"], "png")
                self.assertEqual(params["director_reference_images"], [encoded])
                self.assertEqual(
                    params["director_reference_descriptions"],
                    [
                        {
                            "caption": {"base_caption": "costume", "char_captions": []},
                            "legacy_uc": False,
                        }
                    ],
                )
                self.assertEqual(params["director_reference_strength_values"], [1.0])
                self.assertEqual(params["director_reference_secondary_strength_values"], [0.0])
                self.assertEqual(params["director_reference_information_extracted"], [1.0])
                await router.generate(GenerationRequest(request.model, "cat", ""))
                path, payload = self.requests[-1]
                self.assertEqual(path, "/relay/v1/images/generations")
                self.assertTrue(payload["stream"])
                self.assertNotIn("parameters", payload)

        self.native_status = 201
        images, route = await router.generate(request)
        self.assertEqual((images[0].data, images[0].mime, route), (PNG, "image/png", "primary"))

        self.native_status = 502
        before = len(self.requests)
        with self.assertRaises(GenerationError):
            await router.generate(request)
        self.assertEqual(
            [path for path, _ in self.requests[before:]], ["/relay/v1/ai/generate-image"]
        )

    async def test_precise_route_configuration_and_missing_image(self):
        router = ProviderRouter(
            {
                "primary_base_url": "https://relay.example/v1",
                "primary_api_key": "test",
                "primary_precise_reference": True,
                "secondary_base_url": "https://relay2.example/v1",
                "secondary_api_key": "test",
                "third_base_url": "https://image.novelai.net",
                "third_api_key": "test",
            }
        )
        self.assertEqual(router.precise_reference_routes, ["primary", "third"])
        request = GenerationRequest(
            "nai-diffusion-5-full",
            "cat",
            "",
            precise_reference_type="character",
            precise_reference_id="a" * 64,
        )
        for provider in (router.providers[0], router.providers[2]):
            with self.assertRaisesRegex(GenerationError, "参考图不可用"):
                await provider.generate(request)


if __name__ == "__main__":
    unittest.main()
