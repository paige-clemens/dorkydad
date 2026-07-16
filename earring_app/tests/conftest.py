"""Shared fixtures for earring-maker tests."""
import io

import numpy as np
import pytest
from PIL import Image

from app import app as flask_app


# ---------------------------------------------------------------------------
# Test image helpers
# ---------------------------------------------------------------------------

def _make_png(width: int = 80, height: int = 80, colors: int = 3) -> bytes:
    """Create a simple PNG with *colors* distinct quadrants on a white background."""
    img = Image.new("RGB", (width, height), (255, 255, 255))
    px = img.load()
    palette = [
        (200, 30, 30),   # red
        (30, 30, 200),   # blue
        (30, 180, 30),   # green
        (20, 20, 20),    # black
    ]
    half_w = width // 2
    half_h = height // 2
    for x in range(half_w):
        for y in range(half_h):
            px[x, y] = palette[0 % colors]
    if colors >= 2:
        for x in range(half_w, width):
            for y in range(half_h):
                px[x, y] = palette[1 % colors]
    if colors >= 3:
        for x in range(half_w):
            for y in range(half_h, height):
                px[x, y] = palette[2 % colors]
    if colors >= 4:
        for x in range(half_w, width):
            for y in range(half_h, height):
                px[x, y] = palette[3 % colors]
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_png_with_alpha(width: int = 80, height: int = 80) -> bytes:
    """Create a PNG with an alpha channel (circle foreground)."""
    img = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    px = img.load()
    cx, cy = width // 2, height // 2
    r = min(width, height) // 3
    for x in range(width):
        for y in range(height):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                px[x, y] = (180, 50, 50, 255)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_jpeg(width: int = 80, height: int = 80) -> bytes:
    """Create a simple JPEG test image."""
    img = Image.new("RGB", (width, height), (255, 255, 255))
    px = img.load()
    for x in range(width // 2):
        for y in range(height):
            px[x, y] = (100, 50, 50)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def sample_png():
    return _make_png()


@pytest.fixture
def sample_png_alpha():
    return _make_png_with_alpha()


@pytest.fixture
def sample_jpeg():
    return _make_jpeg()


@pytest.fixture
def sample_png_4_colors():
    return _make_png(colors=4)


def _make_svg() -> bytes:
    """Create a simple SVG test image with colored rectangles."""
    return (
        b'<svg xmlns="http://www.w3.org/2000/svg" width="80" height="80">'
        b'<rect width="40" height="40" fill="#c81e1e"/>'
        b'<rect x="40" width="40" height="40" fill="#1e1ec8"/>'
        b'<rect y="40" width="40" height="40" fill="#1eb41e"/>'
        b'<rect x="40" y="40" width="40" height="40" fill="#141414"/>'
        b'</svg>'
    )


@pytest.fixture
def sample_svg():
    return _make_svg()


# ---------------------------------------------------------------------------
# Flask test client
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path):
    """Flask test client with isolated upload directory."""
    import app as app_module

    flask_app.config["TESTING"] = True
    flask_app.config["SECRET_KEY"] = "test-secret"
    # Override upload dir to tmp
    app_module.UPLOAD_DIR = str(tmp_path / "uploads")

    with flask_app.test_client() as c:
        yield c
