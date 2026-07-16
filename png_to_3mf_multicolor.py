"""
StJacob lion PNG -> multi-color 3MF for Bambu X1C / BambuStudio (AMS).

Output: a single 3MF containing multiple parts:
  - "base"            : full silhouette, 0..BASE_MM           (back color)
  - "color_<idx>_<#hex>": per-color top layer, BASE..THICK    (front colors)

In BambuStudio: import the 3MF -> select all parts -> right-click ->
"Assemble" (or they should already share an object). Then assign each part
to an AMS filament slot using the per-part filament dropdown.

Tweak constants near the top.
"""
import os
import uuid
import zipfile
import numpy as np
from PIL import Image
import cv2
import trimesh
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.affinity import scale as shp_scale
from shapely.validation import make_valid

SRC = "/home/pclemens/Downloads/CleanedUp.jpg"
OUT_3MF = "/home/pclemens/Downloads/StJacob_lion_multicolor.3mf"
PALETTE_PNG = "/home/pclemens/Downloads/StJacob_palette.png"

# Background color of the source image (used when the file has no alpha channel,
# e.g. JPG). Pixels close to this color are treated as background.
BG_RGB = (255, 255, 255)
BG_TOLERANCE = 12         # max per-channel distance from BG_RGB to count as background

TARGET_SIZE_MM   = 39.0   # longest XY side (was 30 mm; +30%)
THICKNESS_MM     = 1.0    # total earring thickness
# Output a MATCHED PAIR of earrings (the second is mirrored left-to-right) so
# it's a "set" for two ears. Set False to output a single earring.
MAKE_PAIR        = True
PAIR_GAP_MM      = 6.0    # gap between the two earrings on the build plate
BASE_MM          = 0.8    # back/base layer thickness; top layer = THICKNESS_MM - BASE_MM
ALPHA_THRESHOLD  = 128
UPSCALE          = 4      # upscale source image NxN before processing (sharper edges)
MIN_REGION_PX    = 400    # drop dust specks smaller than this (in upscaled pixels;
                          # at UPSCALE=4 this is ~25 source pixels of area)
HOLE_DIAMETER_MM = 1.6
HOLE_EDGE_MARGIN = 0.5    # mm clearance between the hole edge and ANY silhouette
                          # edge (top/left/right/bottom of nearest outline).

# Force any pixel darker than this (max RGB channel) into the BLACK slot,
# regardless of nearest-neighbor result. Keeps thin outlines and eyes crisp.
DARK_OVERRIDE_MAX = 70    # 0-255; raise to capture more dark pixels as black
# Dilate the black mask by this many upscaled pixels so outlines don't erode.
BLACK_DILATE_PX = 1

# Inpaint the bright "lens flare" / sparkle inside the mouth before quantizing.
# Any pixel with min(R,G,B) >= REMOVE_FLARE_MIN_RGB AND surrounded by red mouth
# pixels (within REMOVE_FLARE_RADIUS_PX) gets replaced by surrounding context.
REMOVE_FLARE = False
# Two-stage detection: brighter threshold finds the flare *core*; lower
# threshold (with same connected component) captures the rays/halo.
REMOVE_FLARE_CORE_MIN_RGB  = 248  # min(R,G,B) to count as flare core (very bright center)
REMOVE_FLARE_HALO_MIN_RGB  = 130  # min(R,G,B) to count as flare halo/ray
REMOVE_FLARE_DILATE_PX = 6        # final dilation of the flare mask in upscaled px
REMOVE_FLARE_ERODE_PX = 25        # how far inside the silhouette to look for the core

# After labeling, force orange pixels below this Y fraction (0=top, 1=bottom)
# of the silhouette to the peach slot — gives the legs/paws a yellower tone
# than the body, matching the original artwork.
FEET_PEACH_Y_FRAC = 0.55  # tweak; lower => more of the body becomes peach

# Fixed filament palette. Each entry: (name, hex). The image will be quantized
# to exactly these colors (nearest-color match in RGB). Order = AMS filament slot
# (slot 1 = first entry, slot 2 = second, ...). Tweak the hex codes to match
# your actual filaments if they differ.
FIXED_PALETTE = [
    ("black",  "#1a1a1a"),
    ("brown",  "#7a2818"),  # dark maroon-brown (matches mane)
    ("red",    "#e83a18"),  # bright red (matches mouth)
    ("orange", "#f08040"),  # body orange
    ("peach",  "#f5c2a0"),  # belly / body highlights
    ("white",  "#f5f3f2"),
]
N_COLORS = len(FIXED_PALETTE)

# ---------------------------------------------------------------- load + mask
src_im = Image.open(SRC)
has_alpha = src_im.mode in ("RGBA", "LA") or "transparency" in src_im.info
im = src_im.convert("RGBA")
if UPSCALE != 1:
    im = im.resize((im.width * UPSCALE, im.height * UPSCALE), Image.LANCZOS)
W, H = im.size
rgba = np.array(im)
rgb = rgba[:, :, :3].copy()
alpha = rgba[:, :, 3]

if has_alpha:
    silhouette = alpha >= ALPHA_THRESHOLD
else:
    # Build silhouette from background color (e.g., white background JPG).
    bg = np.array(BG_RGB, dtype=np.int32)
    diff = np.abs(rgb.astype(np.int32) - bg).max(axis=2)
    silhouette = diff > BG_TOLERANCE
    # Keep only the largest connected foreground blob (drops stray dust/specks).
    nlbl_s, comp_s, stats_s, _ = cv2.connectedComponentsWithStats(
        silhouette.astype(np.uint8), connectivity=8
    )
    if nlbl_s > 1:
        biggest_s = 1 + int(np.argmax(stats_s[1:, cv2.CC_STAT_AREA]))
        silhouette = (comp_s == biggest_s)
    # Fill internal holes (e.g. white teeth or eye whites accidentally classed
    # as background) by flood-filling from outside the silhouette.
    inv = (~silhouette).astype(np.uint8) * 255
    nlbl_h, comp_h = cv2.connectedComponents(inv, connectivity=4)
    # Mark which background-blobs touch the image border (they are the real BG).
    border_labels = set()
    border_labels.update(np.unique(comp_h[0, :]))
    border_labels.update(np.unique(comp_h[-1, :]))
    border_labels.update(np.unique(comp_h[:, 0]))
    border_labels.update(np.unique(comp_h[:, -1]))
    border_labels.discard(0)  # 0 is foreground (silhouette) here
    interior_bg_mask = (comp_h != 0) & (~np.isin(comp_h, list(border_labels)))
    silhouette = silhouette | interior_bg_mask

# ---------------------------------------------------------------- inpaint lens flare
# The original PNG has a bright "shine" sparkle in the middle of the mouth.
# Detect it (bright pixels surrounded by mouth-red) and inpaint with surrounding
# context so it doesn't dominate the white slot or break the tongue.
if REMOVE_FLARE:
    # Step 1: locate the red mouth (largest connected component of MOUTH-red
    # pixels). Use a saturation gate so we don't include the brown mane.
    R = rgb[:, :, 0].astype(int)
    G = rgb[:, :, 1].astype(int)
    B = rgb[:, :, 2].astype(int)
    # Mouth red: high R, low G/B, AND R clearly stronger than brown (sat. high).
    is_red = (R > 170) & (G < 80) & (B < 80) & ((R - np.maximum(G, B)) > 120) & silhouette
    nlbl_r, comp_r, stats_r, _ = cv2.connectedComponentsWithStats(
        is_red.astype(np.uint8), connectivity=8
    )
    mouth_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    if nlbl_r > 1:
        biggest_r = 1 + int(np.argmax(stats_r[1:, cv2.CC_STAT_AREA]))
        mouth_mask = ((comp_r == biggest_r).astype(np.uint8)) * 255

    if mouth_mask.any():
        # Step 2: bright mask = pixels that are clearly brighter than the
        # surrounding mouth/mane/body. These include the flare core, its rays,
        # and the white teeth/fangs/eye-whites.
        is_bright = (rgb.min(axis=2) >= REMOVE_FLARE_HALO_MIN_RGB) & silhouette
        # Step 3: find the flare seed = brightest pixels near the mouth.
        # The flare sits just OUTSIDE the red tongue (in the dark mouth cavity)
        # so dilate generously to reach it without leaking past black outlines.
        mouth_dilated = cv2.dilate(
            mouth_mask, np.ones((61, 61), np.uint8), iterations=4
        ) & (silhouette.astype(np.uint8) * 255)
        seed = (is_bright.astype(np.uint8) * 255) & mouth_dilated
        if seed.any():
            # Step 4: connected components of *all* bright pixels in the
            # silhouette. The flare's rays are connected to the seed (they
            # radiate continuously). Teeth/fangs/eyes are separated by black
            # outlines so they form their own components.
            nlbl_b, comp_b = cv2.connectedComponents(
                is_bright.astype(np.uint8), connectivity=8
            )
            flare_components = set(np.unique(comp_b[seed > 0])) - {0}
            flare_mask = np.isin(comp_b, list(flare_components)).astype(np.uint8) * 255
            if REMOVE_FLARE_DILATE_PX > 0:
                k = 2 * REMOVE_FLARE_DILATE_PX + 1
                flare_mask = cv2.dilate(flare_mask, np.ones((k, k), np.uint8), iterations=1)
            flare_mask = flare_mask & (silhouette.astype(np.uint8) * 255)
            # Fill with median of real (non-bright) mouth pixels.
            real_mouth = (mouth_mask > 0) & (~is_bright)
            if real_mouth.sum() > 100:
                fill_rgb = np.median(rgb[real_mouth], axis=0).astype(np.uint8)
            else:
                fill_rgb = np.array([200, 30, 25], dtype=np.uint8)
            rgb[flare_mask > 0] = fill_rgb
            Image.fromarray(flare_mask).save("/home/pclemens/Downloads/debug_flare_mask.png")
            Image.fromarray(rgb).save("/home/pclemens/Downloads/debug_inpainted.png")
            print(f"  flare filled with RGB={tuple(int(v) for v in fill_rgb)}: "
                  f"{(flare_mask>0).sum()} px (mouth={(mouth_mask>0).sum()})")

# ---------------------------------------------------------------- quantize colors
# Build palette from FIXED_PALETTE (hex -> RGB). Slot order is preserved so
# labels[i] = i corresponds to AMS slot i+1.
def hex_to_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)], dtype=np.uint8)

palette = np.stack([hex_to_rgb(h) for _name, h in FIXED_PALETTE])
palette_names = [name for name, _h in FIXED_PALETTE]

# Label every silhouette pixel by nearest palette color (Euclidean RGB).
flat_rgb = rgb.reshape(-1, 3).astype(np.int32)
pal = palette.astype(np.int32)
dists = ((flat_rgb[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2)
labels_full = dists.argmin(axis=1).reshape(H, W)
labels = np.where(silhouette, labels_full, -1)

# --- Black-priority pass: force dark pixels into the black slot ----------
black_idx = next(
    (i for i, (n, _h) in enumerate(FIXED_PALETTE) if n.lower() == "black"),
    None,
)
if black_idx is not None:
    dark = silhouette & (rgb.max(axis=2) <= DARK_OVERRIDE_MAX)
    labels = np.where(dark, black_idx, labels)
    if BLACK_DILATE_PX > 0:
        # Dilate the black mask so outlines / eyes don't erode at color borders.
        bm = (labels == black_idx).astype(np.uint8)
        kernel = np.ones((2 * BLACK_DILATE_PX + 1, 2 * BLACK_DILATE_PX + 1), np.uint8)
        bm_dil = cv2.dilate(bm, kernel, iterations=1)
        # Only dilate within the silhouette.
        bm_dil = bm_dil & silhouette.astype(np.uint8)
        labels = np.where(bm_dil > 0, black_idx, labels)

# --- Feet/legs: orange below FEET_PEACH_Y_FRAC -> peach -------------------
orange_idx = next(
    (i for i, (n, _h) in enumerate(FIXED_PALETTE) if n.lower() == "orange"),
    None,
)
peach_idx = next(
    (i for i, (n, _h) in enumerate(FIXED_PALETTE) if n.lower() == "peach"),
    None,
)
if orange_idx is not None and peach_idx is not None and 0.0 < FEET_PEACH_Y_FRAC < 1.0:
    # Bounding box of the silhouette so the threshold is relative to the lion,
    # not the full image canvas (works even with extra transparent padding).
    ys, _xs = np.where(silhouette)
    if len(ys):
        y_min, y_max = int(ys.min()), int(ys.max())
        y_threshold = y_min + int((y_max - y_min) * FEET_PEACH_Y_FRAC)
        below = np.zeros_like(labels, dtype=bool)
        below[y_threshold:, :] = True
        labels = np.where(below & (labels == orange_idx), peach_idx, labels)

# "Base" = largest-area color, used as the back layer.
counts = [int(np.sum(labels == k)) for k in range(N_COLORS)]
base_color_idx = int(np.argmax(counts))

# ---------------------------------------------------------------- save palette swatch
sw_h = 80
sw_w = 120
sw = np.zeros((sw_h, sw_w * N_COLORS, 3), dtype=np.uint8)
for k in range(N_COLORS):
    sw[:, k * sw_w:(k + 1) * sw_w] = palette[k]
sw_img = Image.fromarray(sw)
# annotate
from PIL import ImageDraw
draw = ImageDraw.Draw(sw_img)
for k in range(N_COLORS):
    label = f"slot {k+1}: {palette_names[k]}"
    if k == base_color_idx:
        label += "  (base)"
    txt_color = (0, 0, 0) if palette[k].mean() > 140 else (255, 255, 255)
    draw.text((k * sw_w + 6, 6), label, fill=txt_color)
    draw.text((k * sw_w + 6, sw_h - 18),
              f"#{palette[k,0]:02x}{palette[k,1]:02x}{palette[k,2]:02x}",
              fill=txt_color)
sw_img.save(PALETTE_PNG)

# ---------------------------------------------------------------- contours -> shapely
def mask_to_polygons(bin_mask, clean=True):
    bin_mask = (bin_mask.astype(np.uint8)) * 255
    if clean:
        # Gentle CLOSE only: fills 1-pixel gaps without eroding thin features.
        # No OPEN, since that would kill the thin black outlines.
        k = np.ones((3, 3), np.uint8)
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, k, iterations=1)
    # Use simple chain approximation (collapses collinear pixel-edge points to
    # straight segments) so the resulting meshes don't have a vertex per pixel.
    contours, hierarchy = cv2.findContours(
        bin_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS
    )
    if hierarchy is None:
        return None
    hierarchy = hierarchy[0]
    polys = []
    Hh = bin_mask.shape[0]
    for i, h in enumerate(hierarchy):
        if h[3] != -1:
            continue
        if cv2.contourArea(contours[i]) < MIN_REGION_PX:
            continue
        shell = contours[i].squeeze(1).astype(float)
        if shell.ndim != 2 or len(shell) < 3:
            continue
        shell[:, 1] = Hh - shell[:, 1]  # flip Y
        holes = []
        child = h[2]
        while child != -1:
            if cv2.contourArea(contours[child]) >= MIN_REGION_PX:
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

# Full silhouette polygon
sil_geom = mask_to_polygons(silhouette)
if sil_geom is None:
    raise SystemExit("Empty silhouette")

# Per-color polygons (top layer)
color_geoms = {}
for k in range(N_COLORS):
    g = mask_to_polygons(labels == k)
    if g is not None:
        color_geoms[k] = g

# ---------------------------------------------------------------- scale to mm
minx, miny, maxx, maxy = sil_geom.bounds
px_per_mm = max(maxx - minx, maxy - miny) / TARGET_SIZE_MM
s = 1.0 / px_per_mm

def to_mm(g):
    return shp_scale(g, xfact=s, yfact=s, origin=(0, 0))

# 30 µm simplify tolerance — below 0.2mm nozzle resolution; keeps edges crisp.
SIMPLIFY_MM = 0.03

def simplify(g):
    g2 = g.simplify(SIMPLIFY_MM, preserve_topology=True)
    if not g2.is_valid:
        g2 = make_valid(g2).buffer(0)
    return g2

sil_geom = simplify(to_mm(sil_geom))
color_geoms = {k: simplify(to_mm(g)) for k, g in color_geoms.items()}

# ---------------------------------------------------------------- jump-ring hole
# A point P is a legal hole CENTER iff a disk of (hr + HOLE_EDGE_MARGIN) around
# it lies entirely inside the silhouette. This is equivalent to P being inside
# `sil_geom.buffer(-(hr + HOLE_EDGE_MARGIN))`. We then pick the topmost such P.
minx, miny, maxx, maxy = sil_geom.bounds
hr = HOLE_DIAMETER_MM / 2

eroded = sil_geom.buffer(-(hr + HOLE_EDGE_MARGIN))
if eroded.is_empty:
    raise SystemExit(
        f"Silhouette too thin: cannot fit a {HOLE_DIAMETER_MM} mm hole with "
        f"{HOLE_EDGE_MARGIN} mm margin on all sides."
    )

# Collect all candidate centers and pick the highest one (max y).
def _outline_coords(geom):
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

coords = _outline_coords(eroded)
if len(coords) == 0:
    raise SystemExit("Eroded silhouette has no outline to pick from.")
top_i = int(np.argmax(coords[:, 1]))
hx, hy = float(coords[top_i, 0]), float(coords[top_i, 1])
hole = Point(hx, hy).buffer(hr, quad_segs=48)
print(f"  [hole] center=({hx:.2f}, {hy:.2f}) mm  margin={HOLE_EDGE_MARGIN} mm")

sil_geom = sil_geom.difference(hole)
color_geoms = {k: g.difference(hole) for k, g in color_geoms.items()}

# ---------------------------------------------------------------- extrude
# Base layer (brown) covers the full silhouette at z=0..BASE_MM.
# Top color layers sit on top at z=BASE_MM..THICKNESS with a tiny overlap
# to avoid coplanar faces that hang the slicer.
TOP_OVERLAP_MM = 0.02
TOP_MM = THICKNESS_MM - BASE_MM + TOP_OVERLAP_MM
TOP_Z0 = BASE_MM - TOP_OVERLAP_MM

def extrude(g, height, z0=0.0):
    if g.is_empty:
        return None
    if isinstance(g, Polygon):
        polys = [g]
    else:
        polys = list(g.geoms)
    parts = []
    for p in polys:
        if p.is_empty or p.area < 1e-4:  # drop slivers <0.0001 mm² (~0.3mm edge)
            continue
        try:
            m = trimesh.creation.extrude_polygon(p, height=height)
        except Exception:
            continue
        if z0:
            m.apply_translation([0, 0, z0])
        # Merge duplicate vertices + drop degenerate triangles (no scipy needed).
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

# Base layer covers the full silhouette (z=0..BASE_MM).
base_mesh = extrude(sil_geom, BASE_MM, z0=0.0)
if base_mesh is None:
    raise SystemExit("Failed to extrude base")

# Top color layers sit on the base (z=BASE_MM..THICKNESS).
top_meshes = {}
for k, g in color_geoms.items():
    m = extrude(g, TOP_MM, z0=TOP_Z0)
    if m is not None:
        top_meshes[k] = m

# ---------------------------------------------------------------- center on origin
all_meshes = [base_mesh] + list(top_meshes.values())
combined_bounds = np.array([[m.bounds for m in all_meshes]]).reshape(-1, 2, 3)
xmin = combined_bounds[:, 0, 0].min(); xmax = combined_bounds[:, 1, 0].max()
ymin = combined_bounds[:, 0, 1].min(); ymax = combined_bounds[:, 1, 1].max()
cx = (xmin + xmax) / 2; cy = (ymin + ymax) / 2
for m in all_meshes:
    m.apply_translation([-cx, -cy, 0])

# Tag with vertex colors (visual aid; BambuStudio uses these as a hint)
def tint(mesh, rgb):
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh,
        vertex_colors=np.tile(np.array([rgb[0], rgb[1], rgb[2], 255], dtype=np.uint8),
                              (len(mesh.vertices), 1)),
    )

base_rgb = palette[base_color_idx]
tint(base_mesh, base_rgb)
for k, m in top_meshes.items():
    tint(m, palette[k])

# ---------------------------------------------------------------- export 3MF
# Build a single 3MF where ALL colored parts are components of ONE parent
# object. BambuStudio will then show it as one multi-part object and you can
# assign a filament per part (right column "Filament" dropdown).

def hex_for(rgb):
    return f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"

# Slot = index in FIXED_PALETTE + 1. Base = slot of base_color_idx so the back
# uses the same filament as its matching top region (no extra filament needed).
base_slot = base_color_idx + 1
parts = [(f"base_{palette_names[base_color_idx]}", base_mesh, base_slot)]
# Top layer: one part per non-base color. Skip the base color (the base IS its top).
for k in sorted(top_meshes.keys()):
    if k == base_color_idx:
        continue
    slot = k + 1
    name = f"top_{slot:02d}_{palette_names[k]}"
    parts.append((name, top_meshes[k], slot))


def mesh_xml(mesh):
    v = mesh.vertices
    f = mesh.faces
    # Use formatted floats; mm.
    verts = "".join(
        f'<vertex x="{x:.4f}" y="{y:.4f}" z="{z:.4f}"/>' for x, y, z in v
    )
    tris = "".join(
        f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in f
    )
    return f"<mesh><vertices>{verts}</vertices><triangles>{tris}</triangles></mesh>"


# IDs: 1..N for the leaf part objects, N+1 for the parent assembly object
leaf_ids = list(range(1, len(parts) + 1))
parent_id = len(parts) + 1

leaf_objects = []
for (name, mesh, _slot), oid in zip(parts, leaf_ids):
    leaf_objects.append(
        f'<object id="{oid}" name="{name}" type="model" '
        f'p:UUID="{uuid.uuid4()}">{mesh_xml(mesh)}</object>'
    )

components = "".join(
    f'<component objectid="{oid}"/>' for oid in leaf_ids
)
parent_object = (
    f'<object id="{parent_id}" name="StJacob_lion" type="model" '
    f'p:UUID="{uuid.uuid4()}"><components>{components}</components></object>'
)

build_uuid = uuid.uuid4()
item_uuid = uuid.uuid4()

# Width of the single earring (used to space the matched pair side by side).
earring_width_mm = float(xmax - xmin)
pair_offset = earring_width_mm + PAIR_GAP_MM
# 3MF transform = 4x3 column-major [a b c d e f g h i j k l] applied as:
#   [a d g j]   [x]   [x']
#   [b e h k] * [y] = [y']
#   [c f i l]   [z]   [z']
# (See: https://github.com/3MFConsortium/spec_core)
# Item 1: identity, shifted left so the pair is centered on the build plate.
item1_tx = -pair_offset / 2 if MAKE_PAIR else 0.0
item1_transform = (
    f"1 0 0 0 1 0 0 0 1 {item1_tx:.4f} 0 0"
)
build_items = (
    f'<item objectid="{parent_id}" p:UUID="{item_uuid}" '
    f'transform="{item1_transform}"/>'
)
if MAKE_PAIR:
    # Item 2: mirror X (negate the x-row of the matrix) and shift right.
    item2_tx = pair_offset / 2
    item2_transform = (
        f"-1 0 0 0 1 0 0 0 1 {item2_tx:.4f} 0 0"
    )
    build_items += (
        f'<item objectid="{parent_id}" p:UUID="{uuid.uuid4()}" '
        f'transform="{item2_transform}"/>'
    )

model_xml = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<model unit="millimeter" xml:lang="en-US" '
    'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
    'xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06">'
    '<metadata name="Application">windsurf-script</metadata>'
    '<resources>'
    + "".join(leaf_objects)
    + parent_object
    + '</resources>'
    f'<build p:UUID="{build_uuid}">'
    + build_items +
    '</build>'
    '</model>'
)

# BambuStudio / OrcaSlicer per-part filament assignment lives in
# Metadata/model_settings.config inside the 3MF zip. Format:
#   <config><object id="<parent_id>"><part id="<leaf_id>" subtype="normal_part">
#     <metadata key="name" value="..."/>
#     <metadata key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>
#     <metadata key="extruder" value="N"/>
#   </part>...</object></config>
part_xml_blocks = []
for (name, _mesh, slot), oid in zip(parts, leaf_ids):
    part_xml_blocks.append(
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

MODEL_SETTINGS = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<config>'
    f'<object id="{parent_id}">'
    '<metadata key="name" value="StJacob_lion"/>'
    '<metadata key="extruder" value="1"/>'
    + "".join(part_xml_blocks)
    + '</object></config>'
)

CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
    '<Default Extension="config" ContentType="application/vnd.bambulab-package.3dconfig+xml"/>'
    '</Types>'
)
RELS = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rel-1" Target="/3D/3dmodel.model" '
    'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>'
    '</Relationships>'
)

with zipfile.ZipFile(OUT_3MF, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("[Content_Types].xml", CONTENT_TYPES)
    z.writestr("_rels/.rels", RELS)
    z.writestr("3D/3dmodel.model", model_xml)
    z.writestr("Metadata/model_settings.config", MODEL_SETTINGS)

# Also export individual STLs for users who'd rather load parts manually
parts_dir = "/home/pclemens/Downloads/StJacob_parts"
os.makedirs(parts_dir, exist_ok=True)
# Clear stale STLs from previous runs so the directory only shows current parts.
for _f in os.listdir(parts_dir):
    if _f.endswith(".stl"):
        try:
            os.remove(os.path.join(parts_dir, _f))
        except OSError:
            pass
base_mesh.export(os.path.join(parts_dir, "00_base.stl"))
for k, m in top_meshes.items():
    fname = f"{k+1:02d}_color_#{palette[k,0]:02x}{palette[k,1]:02x}{palette[k,2]:02x}.stl"
    m.export(os.path.join(parts_dir, fname))

print("Filament assignments:")
for name, _m, slot in parts:
    print(f"  slot {slot}: {name}")
print(f"3MF: {OUT_3MF}")
print(f"Palette swatch: {PALETTE_PNG}")
print(f"Per-color STLs: {parts_dir}")
print(f"Base color (idx {base_color_idx}): RGB {tuple(int(v) for v in base_rgb)}")
print(f"Final size: {xmax-xmin:.2f} x {ymax-ymin:.2f} x {THICKNESS_MM:.2f} mm")
