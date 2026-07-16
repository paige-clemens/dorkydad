"""Tests for the Flask web application routes and helpers."""
import io
import json
import os

import pytest
from PIL import Image

from app import _allowed


# ── _allowed helper ────────────────────────────────────────────────────────

class TestAllowed:
    def test_png(self):
        assert _allowed("photo.png") is True

    def test_jpg(self):
        assert _allowed("photo.jpg") is True

    def test_jpeg(self):
        assert _allowed("photo.jpeg") is True

    def test_uppercase(self):
        assert _allowed("PHOTO.PNG") is True

    def test_gif_rejected(self):
        assert _allowed("photo.gif") is False

    def test_no_extension(self):
        assert _allowed("photo") is False

    def test_double_extension(self):
        assert _allowed("photo.tar.png") is True

    def test_svg(self):
        assert _allowed("icon.svg") is True

    def test_svg_uppercase(self):
        assert _allowed("ICON.SVG") is True

    def test_empty_string(self):
        assert _allowed("") is False


# ── Index page ─────────────────────────────────────────────────────────────

class TestIndex:
    def test_get_index(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"Upload" in resp.data or b"upload" in resp.data

    def test_index_contains_form(self, client):
        resp = client.get("/")
        assert b'enctype="multipart/form-data"' in resp.data

    def test_index_has_skip_link(self, client):
        resp = client.get("/")
        assert b"skip-link" in resp.data

    def test_index_has_aria_labels(self, client):
        resp = client.get("/")
        assert b"aria-" in resp.data


# ── Upload route ───────────────────────────────────────────────────────────

class TestUpload:
    def test_upload_no_file(self, client):
        resp = client.post("/upload", follow_redirects=True)
        assert resp.status_code == 200
        assert b"select an image" in resp.data.lower() or b"select" in resp.data.lower()

    def test_upload_empty_filename(self, client):
        data = {"image": (io.BytesIO(b""), "")}
        resp = client.post("/upload", data=data, content_type="multipart/form-data",
                           follow_redirects=True)
        assert resp.status_code == 200

    def test_upload_wrong_type(self, client):
        data = {"image": (io.BytesIO(b"data"), "file.gif")}
        resp = client.post("/upload", data=data, content_type="multipart/form-data",
                           follow_redirects=True)
        assert b"Unsupported" in resp.data or b"unsupported" in resp.data.lower()

    def test_upload_valid_png(self, client, sample_png):
        data = {"image": (io.BytesIO(sample_png), "test.png"), "n_colors": "4"}
        resp = client.post("/upload", data=data, content_type="multipart/form-data")
        assert resp.status_code == 302
        assert "/preview" in resp.headers["Location"]
        assert "n_colors=4" in resp.headers["Location"]

    def test_upload_clamps_n_colors_low(self, client, sample_png):
        data = {"image": (io.BytesIO(sample_png), "test.png"), "n_colors": "0"}
        resp = client.post("/upload", data=data, content_type="multipart/form-data")
        assert resp.status_code == 302
        assert "n_colors=2" in resp.headers["Location"]

    def test_upload_clamps_n_colors_high(self, client, sample_png):
        data = {"image": (io.BytesIO(sample_png), "test.png"), "n_colors": "99"}
        resp = client.post("/upload", data=data, content_type="multipart/form-data")
        assert resp.status_code == 302
        assert "n_colors=16" in resp.headers["Location"]

    def test_upload_invalid_n_colors(self, client, sample_png):
        data = {"image": (io.BytesIO(sample_png), "test.png"), "n_colors": "abc"}
        resp = client.post("/upload", data=data, content_type="multipart/form-data")
        assert resp.status_code == 302
        assert "n_colors=6" in resp.headers["Location"]

    def test_upload_svg(self, client, sample_svg):
        data = {"image": (io.BytesIO(sample_svg), "test.svg"), "n_colors": "4"}
        resp = client.post("/upload", data=data, content_type="multipart/form-data")
        assert resp.status_code == 302
        assert "/preview" in resp.headers["Location"]

    def test_upload_svg_preview_works(self, client, sample_svg):
        data = {"image": (io.BytesIO(sample_svg), "test.svg"), "n_colors": "3"}
        client.post("/upload", data=data, content_type="multipart/form-data")
        resp = client.get("/preview?n_colors=3")
        assert resp.status_code == 200
        assert b"data:image/png;base64," in resp.data

    def test_upload_with_remove_bg(self, client, sample_png):
        data = {
            "image": (io.BytesIO(sample_png), "test.png"),
            "n_colors": "3",
            "remove_bg": "on",
        }
        resp = client.post("/upload", data=data, content_type="multipart/form-data")
        assert resp.status_code == 302
        assert "/preview" in resp.headers["Location"]

    def test_upload_without_remove_bg(self, client, sample_png):
        data = {
            "image": (io.BytesIO(sample_png), "test.png"),
            "n_colors": "3",
        }
        resp = client.post("/upload", data=data, content_type="multipart/form-data")
        assert resp.status_code == 302
        assert "/preview" in resp.headers["Location"]


# ── Preview route ──────────────────────────────────────────────────────────

class TestPreview:
    def _upload(self, client, image_bytes, n_colors=3):
        data = {"image": (io.BytesIO(image_bytes), "test.png"),
                "n_colors": str(n_colors)}
        client.post("/upload", data=data, content_type="multipart/form-data")

    def test_preview_without_upload(self, client):
        resp = client.get("/preview?n_colors=3", follow_redirects=True)
        assert resp.status_code == 200
        # Should flash an error and redirect to index
        assert b"upload" in resp.data.lower()

    def test_preview_after_upload(self, client, sample_png):
        self._upload(client, sample_png, 3)
        resp = client.get("/preview?n_colors=3")
        assert resp.status_code == 200
        assert b"preview" in resp.data.lower() or b"Preview" in resp.data

    def test_preview_contains_base64_image(self, client, sample_png):
        self._upload(client, sample_png, 3)
        resp = client.get("/preview?n_colors=3")
        assert b"data:image/png;base64," in resp.data

    def test_preview_shows_palette(self, client, sample_png):
        self._upload(client, sample_png, 3)
        resp = client.get("/preview?n_colors=3")
        assert b"Slot" in resp.data or b"slot" in resp.data.lower()

    def test_preview_clamps_colors(self, client, sample_png):
        self._upload(client, sample_png)
        resp = client.get("/preview?n_colors=1")  # below minimum
        assert resp.status_code == 200

    def test_preview_has_generate_form(self, client, sample_png):
        self._upload(client, sample_png, 3)
        resp = client.get("/preview?n_colors=3")
        assert b"/generate" in resp.data


# ── Generate route ─────────────────────────────────────────────────────────

class TestGenerate:
    def _upload_and_preview(self, client, image_bytes, n_colors=3):
        data = {"image": (io.BytesIO(image_bytes), "test.png"),
                "n_colors": str(n_colors)}
        client.post("/upload", data=data, content_type="multipart/form-data")
        client.get(f"/preview?n_colors={n_colors}")

    def test_generate_without_session(self, client):
        resp = client.post("/generate", follow_redirects=True)
        assert resp.status_code == 200
        assert b"upload" in resp.data.lower() or b"expired" in resp.data.lower()

    def test_generate_success(self, client, sample_png_4_colors):
        self._upload_and_preview(client, sample_png_4_colors, 3)
        form = {
            "target_size_mm": "30",
            "thickness_mm": "1.0",
            "nub_width_mm": "4",
            "nub_height_mm": "5",
            "hole_diameter_mm": "1.6",
        }
        resp = client.post("/generate", data=form)
        assert resp.status_code == 302
        assert "/download" in resp.headers["Location"]


# ── Download routes ────────────────────────────────────────────────────────

class TestDownload:
    def _full_flow(self, client, image_bytes, n_colors=3):
        data = {"image": (io.BytesIO(image_bytes), "test.png"),
                "n_colors": str(n_colors)}
        client.post("/upload", data=data, content_type="multipart/form-data")
        client.get(f"/preview?n_colors={n_colors}")
        form = {
            "target_size_mm": "30",
            "thickness_mm": "1.0",
            "nub_width_mm": "4",
            "nub_height_mm": "5",
            "hole_diameter_mm": "1.6",
        }
        client.post("/generate", data=form)

    def test_download_page_without_file(self, client):
        resp = client.get("/download", follow_redirects=True)
        assert resp.status_code == 200

    def test_download_page_after_generate(self, client, sample_png_4_colors):
        self._full_flow(client, sample_png_4_colors)
        resp = client.get("/download")
        assert resp.status_code == 200
        assert b"Download" in resp.data

    def test_download_file(self, client, sample_png_4_colors):
        self._full_flow(client, sample_png_4_colors)
        resp = client.get("/download-file")
        assert resp.status_code == 200
        assert resp.content_type is not None
        # Should be a valid zip (3MF)
        assert resp.data[:2] == b"PK"

    def test_download_file_without_generate(self, client):
        resp = client.get("/download-file", follow_redirects=True)
        assert resp.status_code == 200
