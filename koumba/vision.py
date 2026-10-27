"""Small image helper for the optional vision turn.

The photo is read into memory, resized, encoded as JPEG, and sent straight to
the model.  Koumba never writes the image to the data cache or the repository.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image, ImageOps

MAX_IMAGE_SIDE = 1024
JPEG_QUALITY = 85


def image_data_url(image_path: str | Path) -> str:
    """Return a data URL for a resized JPEG version of the image."""

    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source)
        if image.mode in {"RGBA", "LA"} or (
            image.mode == "P" and "transparency" in image.info
        ):
            rgba = image.convert("RGBA")
            background = Image.new("RGB", rgba.size, "white")
            background.paste(rgba, mask=rgba.getchannel("A"))
            image = background
        else:
            image = image.convert("RGB")
        image.thumbnail(
            (MAX_IMAGE_SIDE, MAX_IMAGE_SIDE),
            Image.Resampling.LANCZOS,
        )
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=JPEG_QUALITY, optimize=True)

    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
