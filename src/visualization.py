"""Image processing helpers."""

from PIL import Image


def resize_for_display(image: Image.Image, max_size: int = 512) -> Image.Image:
    """Resize image for display while maintaining aspect ratio."""
    w, h = image.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return image


def ensure_rgb(image: Image.Image) -> Image.Image:
    """Convert image to RGB if necessary."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image
