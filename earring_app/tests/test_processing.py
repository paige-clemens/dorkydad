"""Tests for processing.py — color reduction, silhouette, nub, extrude, 3MF."""
import io
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from PIL import Image
from shapely.geometry import Polygon, MultiPolygon, Point

from processing import (
    is_svg,
    rasterize_svg,
    reduce_colors,
    quantized_to_png_bytes,
    generate_3mf,
    _build_silhouette,
    _mask_to_polygons,
    _extrude,
    _add_nub,
    _outline_coords,
    _mesh_xml,
    _tint,
)


# ── is_svg ─────────────────────────────────────────────────────────────────

class TestIsSvg:
    def test_svg_detected(self, sample_svg):
        assert is_svg(sample_svg) is True

    def test_svg_with_xml_prolog(self):
        data = b'<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg"></svg>'
        assert is_svg(data) is True

    def test_svg_with_leading_whitespace(self):
        data = b'   \n  <svg xmlns="http://www.w3.org/2000/svg"></svg>'
        assert is_svg(data) is True

    def test_png_not_detected(self, sample_png):
        assert is_svg(sample_png) is False

    def test_jpeg_not_detected(self, sample_jpeg):
        assert is_svg(sample_jpeg) is False

    def test_empty_bytes(self):
        assert is_svg(b"") is False


# ── rasterize_svg ──────────────────────────────────────────────────────────

class TestRasterizeSvg:
    def test_returns_png_bytes(self, sample_svg):
        png = rasterize_svg(sample_svg)
        img = Image.open(io.BytesIO(png))
        assert img.format == "PNG"

    def test_output_has_expected_size(self, sample_svg):
        png = rasterize_svg(sample_svg, scale=2.0)
        img = Image.open(io.BytesIO(png))
        # 80×80 SVG at scale=2 → 160×160
        assert img.width == 160
        assert img.height == 160

    def test_default_scale(self, sample_svg):
        png = rasterize_svg(sample_svg)
        img = Image.open(io.BytesIO(png))
        # 80×80 SVG at default scale=4 → 320×320
        assert img.width == 320
        assert img.height == 320

    def test_can_reduce_colors(self, sample_svg):
        png = rasterize_svg(sample_svg)
        quantized, palette = reduce_colors(png, 4)
        assert len(palette) == 4

    def test_can_generate_3mf_from_svg(self, sample_svg):
        png = rasterize_svg(sample_svg)
        palette = [(200, 30, 30), (30, 30, 200), (30, 180, 30)]
        data = generate_3mf(png, palette, upscale=1)
        assert len(data) > 100


# ── reduce_colors ──────────────────────────────────────────────────────────

class TestReduceColors:
    def test_returns_correct_number_of_colors(self, sample_png):
        quantized, palette = reduce_colors(sample_png, 3)
        assert len(palette) == 3

    def test_palette_entries_are_rgb_tuples(self, sample_png):
        _, palette = reduce_colors(sample_png, 4)
        for c in palette:
            assert len(c) == 3
            assert all(0 <= v <= 255 for v in c)

    def test_quantized_shape_matches_input(self, sample_png):
        img = Image.open(io.BytesIO(sample_png))
        quantized, _ = reduce_colors(sample_png, 3)
        assert quantized.shape == (img.height, img.width, 3)

    def test_quantized_uses_only_palette_colors(self, sample_png):
        quantized, palette = reduce_colors(sample_png, 3)
        pal_set = {tuple(c) for c in palette}
        unique = {tuple(row) for row in quantized.reshape(-1, 3)}
        assert unique <= pal_set

    def test_two_colors(self, sample_png):
        quantized, palette = reduce_colors(sample_png, 2)
        assert len(palette) == 2

    def test_works_with_jpeg(self, sample_jpeg):
        quantized, palette = reduce_colors(sample_jpeg, 3)
        assert len(palette) == 3
        assert quantized.dtype == np.uint8


# ── quantized_to_png_bytes ─────────────────────────────────────────────────

class TestQuantizedToPngBytes:
    def test_returns_valid_png(self, sample_png):
        quantized, _ = reduce_colors(sample_png, 3)
        png_bytes = quantized_to_png_bytes(quantized)
        img = Image.open(io.BytesIO(png_bytes))
        assert img.format == "PNG"
        assert img.size == (quantized.shape[1], quantized.shape[0])

    def test_roundtrip_preserves_shape(self, sample_png):
        quantized, _ = reduce_colors(sample_png, 3)
        png_bytes = quantized_to_png_bytes(quantized)
        img = Image.open(io.BytesIO(png_bytes))
        arr = np.array(img)
        assert arr.shape[:2] == quantized.shape[:2]


# ── _build_silhouette ──────────────────────────────────────────────────────

class TestBuildSilhouette:
    def test_alpha_based_silhouette(self, sample_png_alpha):
        img = Image.open(io.BytesIO(sample_png_alpha)).convert("RGBA")
        rgba = np.array(img)
        sil = _build_silhouette(rgba, has_alpha=True)
        assert sil.dtype == bool
        assert sil.any()
        # Corners should be background (transparent)
        assert not sil[0, 0]

    def test_bg_based_silhouette(self, sample_png):
        img = Image.open(io.BytesIO(sample_png)).convert("RGBA")
        rgba = np.array(img)
        sil = _build_silhouette(rgba, has_alpha=False)
        assert sil.dtype == bool
        assert sil.any()

    def test_all_white_returns_empty(self):
        img = Image.new("RGBA", (50, 50), (255, 255, 255, 255))
        rgba = np.array(img)
        sil = _build_silhouette(rgba, has_alpha=False)
        assert not sil.any()


# ── _mask_to_polygons ──────────────────────────────────────────────────────

class TestMaskToPolygons:
    def test_filled_rectangle(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:80, 20:80] = True
        result = _mask_to_polygons(mask, min_region_px=10)
        assert result is not None
        assert result.area > 0

    def test_empty_mask_returns_none(self):
        mask = np.zeros((100, 100), dtype=bool)
        assert _mask_to_polygons(mask) is None

    def test_small_region_filtered(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[0:3, 0:3] = True  # 9 pixels — below default 400 threshold
        assert _mask_to_polygons(mask) is None

    def test_with_hole(self):
        mask = np.zeros((200, 200), dtype=bool)
        mask[20:180, 20:180] = True
        mask[60:140, 60:140] = False  # interior hole
        result = _mask_to_polygons(mask, min_region_px=10)
        assert result is not None


# ── _outline_coords ────────────────────────────────────────────────────────

class TestOutlineCoords:
    def test_simple_polygon(self):
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        coords = _outline_coords(poly)
        assert coords.shape[1] == 2
        assert len(coords) >= 4

    def test_multipolygon(self):
        p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        p2 = Polygon([(5, 5), (6, 5), (6, 6), (5, 6)])
        mp = MultiPolygon([p1, p2])
        coords = _outline_coords(mp)
        assert len(coords) >= 8

    def test_empty_polygon(self):
        poly = Polygon()
        coords = _outline_coords(poly)
        assert coords.shape == (0, 2)


# ── _add_nub ───────────────────────────────────────────────────────────────

class TestAddNub:
    def test_nub_makes_geometry_taller(self):
        square = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
        original_maxy = square.bounds[3]
        combined, hole = _add_nub(square, nub_width_mm=4, nub_height_mm=5, hole_diameter_mm=1.6)
        assert combined.bounds[3] > original_maxy

    def test_hole_is_punched(self):
        square = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
        combined, hole = _add_nub(square, nub_width_mm=4, nub_height_mm=5, hole_diameter_mm=1.6)
        # Hole geometry should exist and be non-empty
        assert not hole.is_empty
        # Combined should have the hole removed (area < square + nub area)
        nub_area_approx = 4 * 5  # rough
        assert combined.area < square.area + nub_area_approx

    def test_hole_inside_combined(self):
        square = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
        combined, hole = _add_nub(square)
        # The hole center should NOT be inside the combined geometry
        assert not combined.contains(hole.centroid)

    def test_custom_dimensions(self):
        square = Polygon([(0, 0), (30, 0), (30, 30), (0, 30)])
        combined, hole = _add_nub(
            square, nub_width_mm=8, nub_height_mm=10, hole_diameter_mm=3.0
        )
        assert combined.bounds[3] > 30


# ── _extrude ───────────────────────────────────────────────────────────────

class TestExtrude:
    def test_simple_polygon(self):
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        mesh = _extrude(poly, height=2.0)
        assert mesh is not None
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0

    def test_z_offset(self):
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        mesh = _extrude(poly, height=1.0, z0=5.0)
        assert mesh.vertices[:, 2].min() >= 4.99

    def test_none_input(self):
        assert _extrude(None, height=1.0) is None

    def test_empty_polygon(self):
        assert _extrude(Polygon(), height=1.0) is None

    def test_multipolygon(self):
        p1 = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        p2 = Polygon([(10, 10), (15, 10), (15, 15), (10, 15)])
        mp = MultiPolygon([p1, p2])
        mesh = _extrude(mp, height=1.0)
        assert mesh is not None
        assert len(mesh.faces) > 0


# ── _mesh_xml ──────────────────────────────────────────────────────────────

class TestMeshXml:
    def test_valid_xml(self):
        import trimesh
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        mesh = trimesh.creation.extrude_polygon(poly, height=1.0)
        xml_str = _mesh_xml(mesh)
        # Should be parseable XML
        root = ET.fromstring(xml_str)
        assert root.tag == "mesh"
        assert root.find("vertices") is not None
        assert root.find("triangles") is not None

    def test_contains_vertex_data(self):
        import trimesh
        poly = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        mesh = trimesh.creation.extrude_polygon(poly, height=1.0)
        xml_str = _mesh_xml(mesh)
        assert "<vertex" in xml_str
        assert "<triangle" in xml_str


# ── _tint ──────────────────────────────────────────────────────────────────

class TestTint:
    def test_applies_vertex_colors(self):
        import trimesh
        poly = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        mesh = trimesh.creation.extrude_polygon(poly, height=1.0)
        _tint(mesh, (255, 0, 0))
        vc = mesh.visual.vertex_colors
        assert vc is not None
        assert vc.shape[1] == 4
        assert np.all(vc[:, 0] == 255)
        assert np.all(vc[:, 3] == 255)


# ── generate_3mf ──────────────────────────────────────────────────────────

class TestGenerate3mf:
    def _default_palette(self):
        return [(20, 20, 20), (200, 30, 30), (30, 30, 200)]

    def test_returns_valid_zip(self, sample_png):
        data = generate_3mf(sample_png, self._default_palette(), upscale=1)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = z.namelist()
            assert "[Content_Types].xml" in names
            assert "_rels/.rels" in names
            assert "3D/3dmodel.model" in names
            assert "Metadata/model_settings.config" in names

    def test_3mf_model_is_valid_xml(self, sample_png):
        data = generate_3mf(sample_png, self._default_palette(), upscale=1)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            model = z.read("3D/3dmodel.model").decode("utf-8")
            root = ET.fromstring(model)
            assert "model" in root.tag

    def test_model_settings_contain_extruder(self, sample_png):
        data = generate_3mf(sample_png, self._default_palette(), upscale=1)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            settings = z.read("Metadata/model_settings.config").decode("utf-8")
            assert "extruder" in settings

    def test_make_pair_false(self, sample_png):
        data = generate_3mf(
            sample_png, self._default_palette(), make_pair=False, upscale=1
        )
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            model = z.read("3D/3dmodel.model").decode("utf-8")
            # Only one <item> in build
            assert model.count("<item ") == 1

    def test_make_pair_true(self, sample_png):
        data = generate_3mf(
            sample_png, self._default_palette(), make_pair=True, upscale=1
        )
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            model = z.read("3D/3dmodel.model").decode("utf-8")
            assert model.count("<item ") == 2

    def test_custom_object_name(self, sample_png):
        data = generate_3mf(
            sample_png, self._default_palette(), object_name="my_earring", upscale=1
        )
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            model = z.read("3D/3dmodel.model").decode("utf-8")
            assert "my_earring" in model

    def test_with_alpha_image(self, sample_png_alpha):
        palette = [(180, 50, 50), (255, 255, 255)]
        data = generate_3mf(sample_png_alpha, palette, upscale=1)
        assert len(data) > 100

    def test_with_jpeg(self, sample_jpeg):
        palette = [(100, 50, 50), (250, 250, 250)]
        data = generate_3mf(sample_jpeg, palette, upscale=1)
        assert len(data) > 100

    def test_no_black_in_palette(self, sample_png):
        # Palette without any very dark color — skip black-priority pass
        palette = [(200, 30, 30), (30, 30, 200), (30, 180, 30)]
        data = generate_3mf(sample_png, palette, upscale=1)
        assert len(data) > 100

    def test_custom_dimensions(self, sample_png):
        data = generate_3mf(
            sample_png,
            self._default_palette(),
            target_size_mm=20.0,
            thickness_mm=2.0,
            base_mm=1.0,
            nub_width_mm=6.0,
            nub_height_mm=8.0,
            hole_diameter_mm=2.0,
            upscale=1,
        )
        assert len(data) > 100

    def test_empty_image_raises(self):
        # All white image → empty silhouette → should raise
        img = Image.new("RGB", (50, 50), (255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        with pytest.raises(ValueError, match="Empty silhouette"):
            generate_3mf(buf.getvalue(), [(200, 0, 0), (0, 0, 200)], upscale=1)
