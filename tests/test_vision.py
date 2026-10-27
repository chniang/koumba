"""Vision helpers resize photos and send them in OpenAI's image format."""

from __future__ import annotations

import base64
import io
from types import SimpleNamespace

from PIL import Image, ImageDraw

from koumba.agent import Agent
from koumba.vision import MAX_IMAGE_SIDE, image_data_url


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def complete(
        self,
        messages,
        tools=None,
        tool_choice="auto",
        temperature=0.2,
    ):
        self.calls.append({"messages": [dict(message) for message in messages]})
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Waaw, Tata 38.",
                        tool_calls=[],
                    )
                )
            ]
        )


def _synthetic_bus_photo(path) -> None:
    image = Image.new("RGB", (2400, 1200), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((300, 300, 2100, 900), outline="black", width=12)
    draw.text((700, 560), "TATA 38", fill="black")
    image.save(path)


def test_image_is_resized_and_encoded_as_jpeg(tmp_path) -> None:
    path = tmp_path / "bus.png"
    _synthetic_bus_photo(path)

    data_url = image_data_url(path)

    assert data_url.startswith("data:image/jpeg;base64,")
    raw = base64.b64decode(data_url.split(",", 1)[1])
    with Image.open(io.BytesIO(raw)) as image:
        assert image.format == "JPEG"
        assert max(image.size) <= MAX_IMAGE_SIDE
        assert image.size[0] > image.size[1]


def test_agent_sends_text_then_the_image_content_part(
    engine,
    tmp_path,
) -> None:
    path = tmp_path / "bus.jpg"
    _synthetic_bus_photo(path)
    client = _FakeClient()
    agent = Agent(client, engine)

    result = agent.run("Is this my bus?", image_path=str(path))

    content = client.calls[0]["messages"][-1]["content"]
    assert content[0] == {"type": "text", "text": "Is this my bus?"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert result.reply == "Waaw, Tata 38."
