import io

import httpx
import pytest_asyncio

import main
from core.container import ServiceContainer


def _png_bytes(size: tuple[int, int] = (4, 4)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, color=(10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


async def test_caption_image_success(client, fake_brain_bundle):
    resp = await client.post(
        "/v1/vision/caption",
        files={"image": ("photo.png", _png_bytes(), "image/png")},
    )

    assert resp.status_code == 200
    assert resp.json() == {"caption": fake_brain_bundle.caption.caption_to_return}

    # The upload reaches CaptionService as the encoded bytes it arrived as --
    # `VisionPipeline.ingest()` does its own decoding, EXIF rotation included.
    assert fake_brain_bundle.caption.calls[-1] == _png_bytes()


async def test_caption_image_marks_the_llm_gate_active(client, fake_brain_bundle):
    """Same reasoning as /emotion-vad: this request precedes the chat turn it
    feeds, so background LLM work should already be standing down."""
    gate = fake_brain_bundle.llm_gate
    assert gate._spoken.is_set() is True  # nothing owed yet, per __init__

    await client.post(
        "/v1/vision/caption",
        files={"image": ("photo.png", _png_bytes(), "image/png")},
    )

    assert gate._spoken.is_set() is False


async def test_caption_image_rejects_empty_upload(client):
    resp = await client.post(
        "/v1/vision/caption",
        files={"image": ("empty.png", b"", "image/png")},
    )
    assert resp.status_code == 400


async def test_caption_image_rejects_unreadable_bytes(client):
    resp = await client.post(
        "/v1/vision/caption",
        files={"image": ("not-an-image.png", b"this is not a real image file", "image/png")},
    )
    assert resp.status_code == 422


@pytest_asyncio.fixture
async def client_vision_disabled(monkeypatch, fake_brain_bundle):
    """A second app instance whose container has no CaptionService at all --
    the same shape `vision_captioning_enabled=False` produces in production.
    """
    fake_brain_bundle.caption = None

    async def _fake_create(cls, settings):
        return fake_brain_bundle.build_container()

    monkeypatch.setattr(ServiceContainer, "create", classmethod(_fake_create))

    async with main.app.router.lifespan_context(main.app):
        transport = httpx.ASGITransport(app=main.app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
            yield ac


async def test_caption_image_disabled_degrades_to_empty_caption(client_vision_disabled):
    resp = await client_vision_disabled.post(
        "/v1/vision/caption",
        files={"image": ("photo.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    assert resp.json() == {"caption": ""}
