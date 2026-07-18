"""
Core image processing and 3MF generation for multicolor earring maker.

Refactored from png_to_3mf_multicolor.py into reusable functions.
"""
import io
import os
from functools import lru_cache
from urllib.request import urlretrieve
import uuid
import zipfile

import cairosvg
import cv2
import numpy as np
import trimesh
from PIL import Image
from shapely.affinity import scale as shp_scale
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.validation import make_valid


# ---------------------------------------------------------------------------
# SVG rasterization
# ---------------------------------------------------------------------------

def rasterize_svg(svg_bytes: bytes, scale: float = 4.0) -> bytes:
    """
    Rasterize SVG data to PNG bytes.

    Parameters
    ----------
    svg_bytes : raw SVG file content
    scale     : render scale factor (higher = more detail)

    Returns
    -------
    bytes – PNG image data
    """
    return cairosvg.svg2png(bytestring=svg_bytes, scale=scale)


def is_svg(data: bytes) -> bool:
    """Detect whether *data* looks like an SVG file."""
    head = data[:512].lstrip()
    return head[:5] == b"<?xml" or head[:4] == b"<svg" or b"<svg" in head[:512]


# ---------------------------------------------------------------------------
# Background removal
# ---------------------------------------------------------------------------

def remove_background(image_bytes: bytes, tolerance: int = 20) -> bytes:
    """
    Remove the background from a raster image.

    Uses flood-fill from the image corners to detect the background color,
    then sets those pixels to transparent. Returns PNG bytes with alpha channel.

    Parameters
    ----------
    image_bytes : raw PNG/JPEG data
    tolerance   : max per-channel distance from corner color to count as bg

    Returns
    -------
    bytes – PNG with transparent background
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    rgba = np.array(img)
    rgb = rgba[:, :, :3]
    H, W = rgba.shape[:2]

    # Sample corner pixels to determine background color
    corners = [rgb[0, 0], rgb[0, W - 1], rgb[H - 1, 0], rgb[H - 1, W - 1]]
    bg_color = np.median(corners, axis=0).astype(np.int32)

    # Build background mask: pixels close to bg_color
    diff = np.abs(rgb.astype(np.int32) - bg_color).max(axis=2)
    bg_mask = diff <= tolerance

    # Flood-fill from edges only — keep interior "bg-colored" pixels as foreground
    seed_mask = np.zeros((H, W), dtype=np.uint8)
    seed_mask[bg_mask] = 255
    # Only keep bg regions connected to the border
    border_seed = np.zeros((H + 2, W + 2), dtype=np.uint8)
    flood_result = seed_mask.copy()
    # Flood fill from each border pixel that is background
    for y in range(H):
        for x in [0, W - 1]:
            if seed_mask[y, x] == 255:
                cv2.floodFill(flood_result, border_seed, (x, y), 128,
                              loDiff=(0,), upDiff=(0,))
    for x in range(W):
        for y in [0, H - 1]:
            if seed_mask[y, x] == 255:
                cv2.floodFill(flood_result, border_seed, (x, y), 128,
                              loDiff=(0,), upDiff=(0,))

    # Background = pixels that were flood-filled (value 128)
    final_bg = flood_result == 128
    rgba[final_bg, 3] = 0

    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG")
    return buf.getvalue()


AI_BACKGROUND_MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
AI_BACKGROUND_MODEL_NAME = "u2netp.onnx"
AI_BACKGROUND_MODEL_SIZE = 320


def _ai_background_model_path() -> str:
    configured_path = os.environ.get("AI_BACKGROUND_MODEL_PATH")
    if configured_path:
        if not os.path.isfile(configured_path):
            raise FileNotFoundError(f"AI background model not found: {configured_path}")
        return configured_path

    model_dir = os.path.join(os.path.expanduser("~"), ".cache", "earring-maker")
    model_path = os.path.join(model_dir, AI_BACKGROUND_MODEL_NAME)
    if not os.path.isfile(model_path):
        os.makedirs(model_dir, exist_ok=True)
        urlretrieve(AI_BACKGROUND_MODEL_URL, model_path)
    return model_path


@lru_cache(maxsize=1)
def _ai_background_session():
    import onnxruntime

    return onnxruntime.InferenceSession(
        _ai_background_model_path(),
        providers=["CPUExecutionProvider"],
    )


def remove_background_ai(image_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    rgba = np.array(img)
    rgb = rgba[:, :, :3].astype(np.float32) / 255.0
    h, w = rgb.shape[:2]

    model_input = cv2.resize(
        rgb,
        (AI_BACKGROUND_MODEL_SIZE, AI_BACKGROUND_MODEL_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    model_input = ((model_input - mean) / std).transpose(2, 0, 1)[np.newaxis, :]
    session = _ai_background_session()
    input_name = session.get_inputs()[0].name
    prediction = session.run(None, {input_name: model_input.astype(np.float32)})[0][0, 0]
    prediction -= prediction.min()
    prediction /= prediction.max() + 1e-8
    alpha = cv2.resize(prediction, (w, h), interpolation=cv2.INTER_LINEAR)
    rgba[:, :, 3] = (rgba[:, :, 3].astype(np.float32) * alpha).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Color reduction / quantization
# ---------------------------------------------------------------------------

def reduce_colors(image_bytes: bytes, n_colors: int) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """
    Reduce an image to *n_colors* using k-means quantization.

    Returns
    -------
    quantized_rgb : np.ndarray  (H, W, 3) uint8 – the quantized image
    palette       : list of (R, G, B) tuples – the palette used
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(img, dtype=np.float32).reshape(-1, 3)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    # 10 attempts with k-means++ initialization produces the best palette
    # quality. The 512 px upload cap keeps the pixel count low enough that
    # this stays responsive.
    _, labels, centers = cv2.kmeans(
        arr, n_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()].reshape(img.size[1], img.size[0], 3)
    palette = [tuple(int(c) for c in row) for row in centers]
    return quantized, palette


def quantized_to_png_bytes(quantized: np.ndarray) -> bytes:
    """Encode a quantized (H,W,3) uint8 array to PNG bytes."""
    buf = io.BytesIO()
    Image.fromarray(quantized).save(buf, format="PNG")
    return buf.getvalue()


def generate_shape_preview(
    image_bytes: bytes,
    palette: list[tuple[int, int, int]] | None = None,
    *,
    raw_bytes: bytes | None = None,
    nub_width_mm: float = 4.0,
    nub_height_mm: float = 5.0,
    hole_diameter_mm: float = 1.6,
    fillet_mm: float = 1.0,
    target_size_mm: float = 39.0,
    checker_size: int = 10,
    dark_override_max: int = 70,
    black_dilate_px: int = 1,
    nub_position: str = "center",
    nub_offset_mm: float = 0.0,
) -> bytes:
    """
    Render a top-down preview that matches the 3MF output: each foreground
    pixel is snapped to its nearest palette color (with black-priority pass),
    the nub is rendered with the base color and smooth fillet, and the
    background is a checkered transparency pattern.  Returns PNG bytes.

    Parameters
    ----------
    image_bytes      : quantized/color-reduced image (used for pixel colors)
    palette          : list of (R,G,B) tuples – palette from color reduction
    raw_bytes        : original image with alpha (for silhouette detection);
                       if None, image_bytes is used for both
    nub_width_mm     : nub width
    nub_height_mm    : nub height
    hole_diameter_mm : hole diameter
    fillet_mm        : fillet radius for smooth nub-silhouette junction
    target_size_mm   : earring target size (for scaling nub proportionally)
    checker_size     : pixel size of the checker squares
    dark_override_max : max channel value to treat as "dark" for black override
    black_dilate_px  : dilation radius for black-priority pass
    nub_position     : "left", "center", or "right" horizontal nub placement
    nub_offset_mm    : signed horizontal offset (mm) from anchor point
    """
    # Load the color image (quantized)
    color_img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    W, H = color_img.size
    color_rgba = np.array(color_img)

    # Load the raw image for silhouette (has alpha from bg removal).
    # Cap silhouette resolution so Shapely contour extraction stays fast on
    # large uploads while the quantized preview image stays crisp.
    MAX_PREVIEW_DIMENSION = 512
    if raw_bytes is not None:
        sil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGBA")
        if max(sil_img.width, sil_img.height) > MAX_PREVIEW_DIMENSION:
            sil_img.thumbnail((MAX_PREVIEW_DIMENSION, MAX_PREVIEW_DIMENSION), Image.LANCZOS)
        if sil_img.size != (W, H):
            sil_img = sil_img.resize((W, H), Image.LANCZOS)
        sil_rgba = np.array(sil_img)
    else:
        sil_rgba = color_rgba

    # Build silhouette from the raw (alpha-bearing) image
    silhouette = _build_silhouette(sil_rgba, has_alpha=True)

    # ── Palette-snap every pixel (mirrors generate_3mf logic) ──
    rgb = color_rgba[:, :, :3].copy()
    base_color = np.array([200, 200, 200], dtype=np.uint8)  # fallback
    if palette and len(palette) > 0 and silhouette.any():
        pal_arr = np.array(palette, dtype=np.uint8)
        pal_i32 = pal_arr.astype(np.int32)
        n_colors = len(palette)

        # Nearest-color quantisation
        flat_rgb = rgb.reshape(-1, 3).astype(np.int32)
        dists = ((flat_rgb[:, None, :] - pal_i32[None, :, :]) ** 2).sum(axis=2)
        labels_full = dists.argmin(axis=1).reshape(H, W)
        labels = np.where(silhouette, labels_full, -1)

        # Black-priority pass (same as generate_3mf)
        black_idx = None
        darkest_brightness = 999
        for i, c in enumerate(palette):
            brightness = max(c)
            if brightness < darkest_brightness:
                darkest_brightness = brightness
                darkest_idx = i
        if darkest_brightness < 80:
            black_idx = darkest_idx

        if black_idx is not None:
            dark = silhouette & (rgb.max(axis=2) <= dark_override_max)
            labels = np.where(dark, black_idx, labels)
            if black_dilate_px > 0:
                bm = (labels == black_idx).astype(np.uint8)
                kernel = np.ones(
                    (2 * black_dilate_px + 1, 2 * black_dilate_px + 1), np.uint8)
                bm_dil = cv2.dilate(bm, kernel, iterations=1)
                bm_dil = bm_dil & silhouette.astype(np.uint8)
                labels = np.where(bm_dil > 0, black_idx, labels)

        # Base color = largest area
        counts = [int(np.sum(labels == k)) for k in range(n_colors)]
        base_color_idx = int(np.argmax(counts))
        base_color = pal_arr[base_color_idx]

        # Paint each pixel with its palette color
        snapped = np.zeros_like(rgb)
        for k in range(n_colors):
            snapped[labels == k] = pal_arr[k]
        rgb = snapped

    # ── Nub geometry via Shapely (matches generate_3mf) ──
    sil_geom = _mask_to_polygons(silhouette)
    if sil_geom is None:
        checker = _make_checker(H, W, checker_size)
        output = np.where(silhouette[:, :, None], rgb, checker)
        buf = io.BytesIO()
        Image.fromarray(output.astype(np.uint8)).save(buf, format="PNG")
        return buf.getvalue()

    minx, miny, maxx, maxy = sil_geom.bounds
    px_extent = max(maxx - minx, maxy - miny)
    px_per_mm = px_extent / target_size_mm
    s = 1.0 / px_per_mm

    sil_mm = shp_scale(sil_geom, xfact=s, yfact=s, origin=(0, 0))
    combined_mm, hole_mm = _add_nub(
        sil_mm,
        nub_width_mm=nub_width_mm,
        nub_height_mm=nub_height_mm,
        hole_diameter_mm=hole_diameter_mm,
        fillet_mm=fillet_mm,
        nub_position=nub_position,
        nub_offset_mm=nub_offset_mm,
    )
    combined_px = shp_scale(combined_mm, xfact=px_per_mm, yfact=px_per_mm, origin=(0, 0))

    # Rasterize Shapely geometry to pixel mask
    cminx, cminy, cmaxx, cmaxy = combined_px.bounds
    extra_top_px = int(max(0, cmaxy - H))
    new_H = H + extra_top_px

    def _geom_to_pixel_coords(geom, img_h):
        if geom.geom_type == 'Polygon':
            geom_list = [geom]
        elif geom.geom_type == 'MultiPolygon':
            geom_list = list(geom.geoms)
        else:
            return [], []
        shells, holes = [], []
        for poly in geom_list:
            ext = np.array(poly.exterior.coords)
            pts = np.column_stack([ext[:, 0], img_h - ext[:, 1]]).astype(np.int32)
            shells.append(pts)
            for interior in poly.interiors:
                hpts = np.array(interior.coords)
                hpts = np.column_stack([hpts[:, 0], img_h - hpts[:, 1]]).astype(np.int32)
                holes.append(hpts)
        return shells, holes

    combined_mask = np.zeros((new_H, W), dtype=np.uint8)
    shells, holes = _geom_to_pixel_coords(combined_px, new_H)
    if shells:
        cv2.fillPoly(combined_mask, shells, 255)
    if holes:
        cv2.fillPoly(combined_mask, holes, 0)
    combined_mask_bool = combined_mask > 0

    # Extend canvases for nub padding
    if extra_top_px > 0:
        rgb = np.vstack([np.zeros((extra_top_px, W, 3), dtype=np.uint8), rgb])
        silhouette = np.vstack([
            np.zeros((extra_top_px, W), dtype=bool), silhouette
        ])

    # Composite: palette-snapped colors for silhouette, base color for nub,
    # checkered pattern everywhere else
    checker = _make_checker(new_H, W, checker_size)
    nub_only = combined_mask_bool & ~silhouette
    rgb[nub_only] = base_color
    output = np.where(combined_mask_bool[:, :, None], rgb, checker)

    buf = io.BytesIO()
    Image.fromarray(output.astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _make_checker(H: int, W: int, size: int = 10) -> np.ndarray:
    """Generate a checkered pattern (light gray / white) as (H, W, 3) uint8."""
    ys = np.arange(H) // size
    xs = np.arange(W) // size
    grid = (ys[:, None] + xs[None, :]) % 2  # 0 or 1
    checker = np.where(grid[:, :, None] == 0,
                       np.array([220, 220, 220], dtype=np.uint8),
                       np.array([255, 255, 255], dtype=np.uint8))
    return checker


# ---------------------------------------------------------------------------
# Silhouette extraction
# ---------------------------------------------------------------------------

_BG_RGB = (255, 255, 255)
_BG_TOLERANCE = 12
_ALPHA_THRESHOLD = 128
_MIN_REGION_PX = 400


def _build_silhouette(rgba: np.ndarray, has_alpha: bool) -> np.ndarray:
    """Return a boolean mask of the foreground silhouette."""
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3]

    if has_alpha:
        silhouette = alpha >= _ALPHA_THRESHOLD
    else:
        bg = np.array(_BG_RGB, dtype=np.int32)
        diff = np.abs(rgb.astype(np.int32) - bg).max(axis=2)
        silhouette = diff > _BG_TOLERANCE
        # Keep only the largest connected foreground blob
        nlbl, comp, stats, _ = cv2.connectedComponentsWithStats(
            silhouette.astype(np.uint8), connectivity=8
        )
        if nlbl > 1:
            biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            silhouette = comp == biggest
        # Fill interior holes
        inv = (~silhouette).astype(np.uint8) * 255
        nlbl_h, comp_h = cv2.connectedComponents(inv, connectivity=4)
        border_labels = set()
        border_labels.update(np.unique(comp_h[0, :]))
        border_labels.update(np.unique(comp_h[-1, :]))
        border_labels.update(np.unique(comp_h[:, 0]))
        border_labels.update(np.unique(comp_h[:, -1]))
        border_labels.discard(0)
        interior_bg = (comp_h != 0) & (~np.isin(comp_h, list(border_labels)))
        silhouette = silhouette | interior_bg

    return silhouette


# ---------------------------------------------------------------------------
# Contour → Shapely polygons
# ---------------------------------------------------------------------------

def _mask_to_polygons(bin_mask: np.ndarray, min_region_px: int = _MIN_REGION_PX):
    """Convert a boolean mask to Shapely polygon(s)."""
    bm = (bin_mask.astype(np.uint8)) * 255
    k = np.ones((3, 3), np.uint8)
    bm = cv2.morphologyEx(bm, cv2.MORPH_CLOSE, k, iterations=1)
    contours, hierarchy = cv2.findContours(
        bm, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS
    )
    if hierarchy is None:
        return None
    hierarchy = hierarchy[0]
    Hh = bm.shape[0]
    polys = []
    for i, h in enumerate(hierarchy):
        if h[3] != -1:
            continue
        if cv2.contourArea(contours[i]) < min_region_px:
            continue
        shell = contours[i].squeeze(1).astype(float)
        if shell.ndim != 2 or len(shell) < 3:
            continue
        shell[:, 1] = Hh - shell[:, 1]
        holes = []
        child = h[2]
        while child != -1:
            if cv2.contourArea(contours[child]) >= min_region_px:
                hh = contours[child].squeeze(1).astype(float)
                if hh.ndim == 2 and len(hh) >= 3:
                    hh[:, 1] = Hh - hh[:, 1]
                    holes.append(hh)
            child = hierarchy[child][0]
        p = Polygon(shell, holes)
        if not p.is_valid:
            p = make_valid(p).buffer(0)
        if p.is_empty:
            continue
        if isinstance(p, MultiPolygon):
            polys.extend(list(p.geoms))
        else:
            polys.append(p)
    if not polys:
        return None
    return polys[0] if len(polys) == 1 else MultiPolygon(polys)


# ---------------------------------------------------------------------------
# Extrude polygon to mesh
# ---------------------------------------------------------------------------

def _extrude(g, height: float, z0: float = 0.0):
    """Extrude a Shapely geometry to a trimesh."""
    if g is None or g.is_empty:
        return None
    polys = list(g.geoms) if hasattr(g, "geoms") else [g]
    parts = []
    for p in polys:
        if p.is_empty or p.area < 1e-4:
            continue
        try:
            m = trimesh.creation.extrude_polygon(p, height=height)
        except Exception:
            continue
        if z0:
            m.apply_translation([0, 0, z0])
        m.merge_vertices()
        nondeg = (
            (m.faces[:, 0] != m.faces[:, 1])
            & (m.faces[:, 1] != m.faces[:, 2])
            & (m.faces[:, 0] != m.faces[:, 2])
        )
        if not nondeg.all():
            m.update_faces(nondeg)
        if len(m.faces) == 0:
            continue
        parts.append(m)
    if not parts:
        return None
    return trimesh.util.concatenate(parts)


# ---------------------------------------------------------------------------
# Nub (tab) with hole for earring hanging
# ---------------------------------------------------------------------------

def _add_nub(sil_geom, nub_width_mm: float = 4.0, nub_height_mm: float = 5.0,
             hole_diameter_mm: float = 1.6, fillet_mm: float = 1.0,
             nub_position: str = "center", nub_offset_mm: float = 0.0):
    """
    Add a nub at the top of the silhouette, smoothly blended into the
    silhouette contour with a fillet so the junction is seamless even on
    irregular shapes. Punch a hole through it for earring attachment.

    nub_position controls horizontal anchor point:
      "center" – topmost point (default)
      "left"   – leftmost among the topmost points
      "right"  – rightmost among the topmost points
    nub_offset_mm is a signed horizontal offset (mm) from that anchor point;
      positive moves the nub to the right, negative to the left.  The final
      position is clamped so the nub base stays within the silhouette.

    Returns (modified_sil_geom, hole_geometry).
    """
    # Find the topmost points of the silhouette boundary
    coords = _outline_coords(sil_geom)
    max_y = float(coords[:, 1].max())
    # Points within a small tolerance of the top edge
    top_tol = (max_y - float(coords[:, 1].min())) * 0.02
    top_mask = coords[:, 1] >= (max_y - top_tol)
    top_coords = coords[top_mask]

    if nub_position == "left":
        pick_i = int(np.argmin(top_coords[:, 0]))
    elif nub_position == "right":
        pick_i = int(np.argmax(top_coords[:, 0]))
    else:  # center
        pick_i = int(np.argmax(top_coords[:, 1]))

    attach_x = float(top_coords[pick_i, 0]) + nub_offset_mm
    attach_y = float(top_coords[pick_i, 1])

    # Clamp nub base to silhouette bounding box so it always overlaps
    minx, _, maxx, _ = sil_geom.bounds
    hw = nub_width_mm / 2
    attach_x = max(minx + hw, min(maxx - hw, attach_x))

    # Build the nub stem: a rectangle topped with a semicircle
    hw = nub_width_mm / 2
    nub_bottom = attach_y
    nub_top = attach_y + nub_height_mm

    # Rectangle
    rect = Polygon([
        (attach_x - hw, nub_bottom),
        (attach_x + hw, nub_bottom),
        (attach_x + hw, nub_top - hw),
        (attach_x - hw, nub_top - hw),
    ])
    # Semicircle on top
    semicircle = Point(attach_x, nub_top - hw).buffer(hw, quad_segs=24)
    clip_box = Polygon([
        (attach_x - hw - 1, nub_top - hw),
        (attach_x + hw + 1, nub_top - hw),
        (attach_x + hw + 1, nub_top + 1),
        (attach_x - hw - 1, nub_top + 1),
    ])
    semicircle = semicircle.intersection(clip_box)
    nub = rect.union(semicircle)
    if not nub.is_valid:
        nub = make_valid(nub).buffer(0)

    # Union nub with silhouette, then apply a buffer-unbuffer (fillet) to
    # smooth the junction between nub and silhouette contour.
    combined = sil_geom.union(nub)
    if not combined.is_valid:
        combined = make_valid(combined).buffer(0)

    # Fillet: dilate then erode by the same radius rounds sharp concave
    # corners at the junction between the nub stem and the silhouette.
    if fillet_mm > 0:
        combined = combined.buffer(fillet_mm, join_style="round", quad_segs=16)
        combined = combined.buffer(-fillet_mm, join_style="round", quad_segs=16)
        if not combined.is_valid:
            combined = make_valid(combined).buffer(0)

    # Hole center: in the rounded part of the nub
    hole_cx = attach_x
    hole_cy = nub_top - hw  # center of the semicircle
    hr = hole_diameter_mm / 2
    hole = Point(hole_cx, hole_cy).buffer(hr, quad_segs=48)

    # Punch hole
    combined = combined.difference(hole)

    return combined, hole


def _outline_coords(geom):
    """Extract all outline coordinates from a Shapely geometry."""
    polys = geom.geoms if hasattr(geom, "geoms") else [geom]
    out = []
    for p in polys:
        if p.is_empty:
            continue
        out.append(np.array(p.exterior.coords))
        for ring in p.interiors:
            out.append(np.array(ring.coords))
    if not out:
        return np.empty((0, 2))
    return np.vstack(out)


# ---------------------------------------------------------------------------
# 3MF XML generation
# ---------------------------------------------------------------------------

def _mesh_xml(mesh) -> str:
    v = mesh.vertices
    f = mesh.faces
    verts = "".join(
        f'<vertex x="{x:.4f}" y="{y:.4f}" z="{z:.4f}"/>' for x, y, z in v
    )
    tris = "".join(
        f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in f
    )
    return f"<mesh><vertices>{verts}</vertices><triangles>{tris}</triangles></mesh>"


def _tint(mesh, rgb):
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh,
        vertex_colors=np.tile(
            np.array([rgb[0], rgb[1], rgb[2], 255], dtype=np.uint8),
            (len(mesh.vertices), 1),
        ),
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate_3mf(
    image_bytes: bytes,
    palette: list[tuple[int, int, int]],
    *,
    target_size_mm: float = 39.0,
    thickness_mm: float = 1.0,
    base_mm: float = 0.8,
    upscale: int = 4,
    nub_width_mm: float = 4.0,
    nub_height_mm: float = 5.0,
    hole_diameter_mm: float = 1.6,
    pair_gap_mm: float = 6.0,
    dark_override_max: int = 70,
    black_dilate_px: int = 1,
    object_name: str = "earring",
    nub_position: str = "center",
    nub_offset_mm: float = 0.0,
) -> bytes:
    """
    Full pipeline: image bytes + palette → 3MF file bytes.

    Parameters
    ----------
    image_bytes   : raw uploaded file bytes (PNG or JPEG)
    palette       : list of (R,G,B) tuples – the reduced color palette
    target_size_mm: longest XY dimension of the earring
    thickness_mm  : total Z thickness
    base_mm       : back-layer thickness
    upscale       : upscale factor for source image
    nub_width_mm  : width of the hanging nub
    nub_height_mm : height of the hanging nub
    hole_diameter_mm : diameter of the earring hole
    pair_gap_mm   : gap between the two earring copies
    dark_override_max : pixels darker than this → black slot
    black_dilate_px   : dilate black outlines by this many px
    object_name   : name embedded in the 3MF
    nub_position  : "left", "center", or "right" horizontal nub placement
    nub_offset_mm : signed horizontal offset (mm) from anchor point

    Returns
    -------
    bytes – the 3MF zip file contents
    """
    n_colors = len(palette)
    pal_arr = np.array(palette, dtype=np.uint8)

    # Cap working resolution to keep Shapely/trimesh operations fast.  The final
    # print is only ~39 mm across, so anything above ~2048 px is overkill for
    # 0.4 mm nozzles and causes exponential slowdown in mesh operations.
    MAX_PROCESSED_DIMENSION = 2048

    # Load & upscale
    src_im = Image.open(io.BytesIO(image_bytes))
    has_alpha = src_im.mode in ("RGBA", "LA") or "transparency" in src_im.info
    im = src_im.convert("RGBA")

    effective_max = max(im.width, im.height) * upscale
    if effective_max > MAX_PROCESSED_DIMENSION:
        # Reduce upscale so the processed image stays within the cap.
        scale = MAX_PROCESSED_DIMENSION / effective_max
        new_w = max(1, int(round(im.width * scale)))
        new_h = max(1, int(round(im.height * scale)))
        im = im.resize((new_w, new_h), Image.LANCZOS)
    elif upscale != 1:
        im = im.resize((im.width * upscale, im.height * upscale), Image.LANCZOS)

    W, H = im.size
    rgba = np.array(im)
    rgb = rgba[:, :, :3].copy()

    # Silhouette
    silhouette = _build_silhouette(rgba, has_alpha)

    # Quantize to palette (nearest-color in RGB)
    flat_rgb = rgb.reshape(-1, 3).astype(np.int32)
    pal_i32 = pal_arr.astype(np.int32)
    dists = ((flat_rgb[:, None, :] - pal_i32[None, :, :]) ** 2).sum(axis=2)
    labels_full = dists.argmin(axis=1).reshape(H, W)
    labels = np.where(silhouette, labels_full, -1)

    # Black-priority pass
    black_idx = None
    darkest_idx = None
    darkest_brightness = 999
    for i, c in enumerate(palette):
        brightness = max(c)
        if brightness < darkest_brightness:
            darkest_brightness = brightness
            darkest_idx = i
    if darkest_brightness < 80:
        black_idx = darkest_idx

    if black_idx is not None:
        dark = silhouette & (rgb.max(axis=2) <= dark_override_max)
        labels = np.where(dark, black_idx, labels)
        if black_dilate_px > 0:
            bm = (labels == black_idx).astype(np.uint8)
            kernel = np.ones((2 * black_dilate_px + 1, 2 * black_dilate_px + 1), np.uint8)
            bm_dil = cv2.dilate(bm, kernel, iterations=1)
            bm_dil = bm_dil & silhouette.astype(np.uint8)
            labels = np.where(bm_dil > 0, black_idx, labels)

    # Base color = largest area
    counts = [int(np.sum(labels == k)) for k in range(n_colors)]
    base_color_idx = int(np.argmax(counts))

    # Contours → Shapely
    sil_geom = _mask_to_polygons(silhouette)
    if sil_geom is None:
        raise ValueError("Empty silhouette – cannot generate earring.")

    color_geoms = {}
    for k in range(n_colors):
        g = _mask_to_polygons(labels == k)
        if g is not None:
            color_geoms[k] = g

    # Scale to mm
    minx, miny, maxx, maxy = sil_geom.bounds
    px_per_mm = max(maxx - minx, maxy - miny) / target_size_mm
    s = 1.0 / px_per_mm

    def to_mm(g):
        return shp_scale(g, xfact=s, yfact=s, origin=(0, 0))

    SIMPLIFY_MM = 0.03

    def simplify(g):
        g2 = g.simplify(SIMPLIFY_MM, preserve_topology=True)
        if not g2.is_valid:
            g2 = make_valid(g2).buffer(0)
        return g2

    sil_geom = simplify(to_mm(sil_geom))
    color_geoms = {k: simplify(to_mm(g)) for k, g in color_geoms.items()}

    # Add nub with hole
    sil_geom, hole = _add_nub(
        sil_geom,
        nub_width_mm=nub_width_mm,
        nub_height_mm=nub_height_mm,
        hole_diameter_mm=hole_diameter_mm,
        nub_position=nub_position,
        nub_offset_mm=nub_offset_mm,
    )
    color_geoms = {k: g.difference(hole) for k, g in color_geoms.items()}

    # Extrude
    TOP_OVERLAP_MM = 0.02
    TOP_MM = thickness_mm - base_mm + TOP_OVERLAP_MM
    TOP_Z0 = base_mm - TOP_OVERLAP_MM

    base_mesh = _extrude(sil_geom, base_mm, z0=0.0)
    if base_mesh is None:
        raise ValueError("Failed to extrude base mesh.")

    top_meshes = {}
    for k, g in color_geoms.items():
        m = _extrude(g, TOP_MM, z0=TOP_Z0)
        if m is not None:
            top_meshes[k] = m

    # Center on origin
    all_meshes = [base_mesh] + list(top_meshes.values())
    combined_bounds = np.array([[m.bounds for m in all_meshes]]).reshape(-1, 2, 3)
    xmin = combined_bounds[:, 0, 0].min()
    xmax = combined_bounds[:, 1, 0].max()
    ymin = combined_bounds[:, 0, 1].min()
    ymax = combined_bounds[:, 1, 1].max()
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    for m in all_meshes:
        m.apply_translation([-cx, -cy, 0])

    # Tint meshes
    base_rgb = pal_arr[base_color_idx]
    _tint(base_mesh, base_rgb)
    for k, m in top_meshes.items():
        _tint(m, pal_arr[k])

    # Build 3MF
    palette_names = [f"color_{i+1}" for i in range(n_colors)]
    base_slot = base_color_idx + 1
    parts = [(f"base_{palette_names[base_color_idx]}", base_mesh, base_slot)]
    for k in sorted(top_meshes.keys()):
        if k == base_color_idx:
            continue
        slot = k + 1
        name = f"top_{slot:02d}_{palette_names[k]}"
        parts.append((name, top_meshes[k], slot))

    leaf_ids = list(range(1, len(parts) + 1))
    parent_id_1 = len(parts) + 1
    parent_id_2 = len(parts) + 2

    leaf_objects = []
    for (name, mesh, _slot), oid in zip(parts, leaf_ids):
        leaf_objects.append(
            f'<object id="{oid}" name="{name}" type="model" '
            f'p:UUID="{uuid.uuid4()}">{_mesh_xml(mesh)}</object>'
        )

    components = "".join(
        f'<component objectid="{oid}"/>' for oid in leaf_ids
    )
    parent_object_1 = (
        f'<object id="{parent_id_1}" name="{object_name}_1" type="model" '
        f'p:UUID="{uuid.uuid4()}"><components>{components}</components></object>'
    )
    parent_object_2 = (
        f'<object id="{parent_id_2}" name="{object_name}_2" type="model" '
        f'p:UUID="{uuid.uuid4()}"><components>{components}</components></object>'
    )

    # Place two copies side by side
    earring_width_mm = float(xmax - xmin)
    pair_offset = earring_width_mm + pair_gap_mm
    build_uuid = uuid.uuid4()

    item1_tx = -pair_offset / 2
    item1_transform = f"1 0 0 0 1 0 0 0 1 {item1_tx:.4f} 0 0"
    item2_tx = pair_offset / 2
    item2_transform = f"1 0 0 0 1 0 0 0 1 {item2_tx:.4f} 0 0"

    build_items = (
        f'<item objectid="{parent_id_1}" p:UUID="{uuid.uuid4()}" '
        f'transform="{item1_transform}"/>'
        f'<item objectid="{parent_id_2}" p:UUID="{uuid.uuid4()}" '
        f'transform="{item2_transform}"/>'
    )

    model_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<model unit="millimeter" xml:lang="en-US" '
        'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
        'xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06">'
        '<metadata name="Application">earring-web-app</metadata>'
        '<resources>'
        + "".join(leaf_objects)
        + parent_object_1
        + parent_object_2
        + '</resources>'
        f'<build p:UUID="{build_uuid}">'
        + build_items
        + '</build>'
        '</model>'
    )

    def _part_xml_blocks_for(parent_name):
        blocks = []
        for (name, _mesh, slot), oid in zip(parts, leaf_ids):
            blocks.append(
                f'<part id="{oid}" subtype="normal_part">'
                f'<metadata key="name" value="{name}"/>'
                f'<metadata key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>'
                f'<metadata key="source_file" value=""/>'
                f'<metadata key="source_object_id" value="0"/>'
                f'<metadata key="source_volume_id" value="{oid - 1}"/>'
                f'<metadata key="source_offset_x" value="0"/>'
                f'<metadata key="source_offset_y" value="0"/>'
                f'<metadata key="source_offset_z" value="0"/>'
                f'<metadata key="extruder" value="{slot}"/>'
                f'</part>'
            )
        return blocks

    model_settings = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<config>'
        f'<object id="{parent_id_1}">'
        f'<metadata key="name" value="{object_name}_1"/>'
        '<metadata key="extruder" value="1"/>'
        + "".join(_part_xml_blocks_for(f"{object_name}_1"))
        + '</object>'
        f'<object id="{parent_id_2}">'
        f'<metadata key="name" value="{object_name}_2"/>'
        '<metadata key="extruder" value="1"/>'
        + "".join(_part_xml_blocks_for(f"{object_name}_2"))
        + '</object></config>'
    )

    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType='
        '"application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="model" ContentType='
        '"application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
        '<Default Extension="config" ContentType='
        '"application/vnd.bambulab-package.3dconfig+xml"/>'
        '</Types>'
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rel-1" Target="/3D/3dmodel.model" '
        'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>'
        '</Relationships>'
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("3D/3dmodel.model", model_xml)
        z.writestr("Metadata/model_settings.config", model_settings)
    return buf.getvalue()
