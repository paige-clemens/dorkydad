"""
Convert StJacob_new_logo PNG -> flat extruded STL for 3D-printed earrings.

- Uses the alpha channel to get the lion silhouette.
- Builds polygons (with holes) from contours.
- Extrudes to a given thickness (mm) and scales so the longest side is `target_size_mm`.
"""
import numpy as np
from PIL import Image
import trimesh
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.affinity import scale as shp_scale, translate as shp_translate
from shapely.validation import make_valid
import cv2

SRC = "/home/pclemens/Downloads/StJacob_new_logo__1_.png"
DST = "/home/pclemens/Downloads/StJacob_new_logo.stl"

THICKNESS_MM = 1.0          # earring thickness
TARGET_SIZE_MM = 30.0       # longest side of the earring in mm
ALPHA_THRESHOLD = 128       # pixel is "solid" if alpha >= this
HOLE_DIAMETER_MM = 1.6      # jump-ring hole diameter
HOLE_EDGE_MARGIN_MM = 1.2   # distance from top edge of silhouette to hole edge

# 1) Load image, get binary mask from alpha
im = Image.open(SRC).convert("RGBA")
alpha = np.array(im.split()[-1])
mask = (alpha >= ALPHA_THRESHOLD).astype(np.uint8) * 255

# Optional: close tiny gaps so the silhouette is one solid piece
kernel = np.ones((3, 3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

# 2) Find contours with hierarchy (so we get holes)
contours, hierarchy = cv2.findContours(
    mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS
)
if hierarchy is None:
    raise SystemExit("No contours found.")

hierarchy = hierarchy[0]  # shape (N,4): [next, prev, first_child, parent]

H = mask.shape[0]

def to_xy(cnt):
    pts = cnt.squeeze(1).astype(float)
    # Flip Y so the model is upright (image y-down -> world y-up)
    pts[:, 1] = H - pts[:, 1]
    return pts

# Build polygons: outer contours (parent == -1) with their child holes.
polys = []
for i, h in enumerate(hierarchy):
    parent = h[3]
    if parent != -1:
        continue
    if cv2.contourArea(contours[i]) < 20:
        continue
    shell = to_xy(contours[i])
    if len(shell) < 3:
        continue
    holes = []
    child = h[2]
    while child != -1:
        if cv2.contourArea(contours[child]) >= 20:
            hole = to_xy(contours[child])
            if len(hole) >= 3:
                holes.append(hole)
        child = hierarchy[child][0]
    poly = Polygon(shell, holes)
    if not poly.is_valid:
        poly = make_valid(poly).buffer(0)
    if poly.is_empty:
        continue
    if isinstance(poly, MultiPolygon):
        polys.extend(list(poly.geoms))
    else:
        polys.append(poly)

if not polys:
    raise SystemExit("No usable polygons.")

geom = polys[0] if len(polys) == 1 else MultiPolygon(polys)

# 3) Scale polygons from pixels to mm (longest side -> TARGET_SIZE_MM)
minx, miny, maxx, maxy = geom.bounds
px_per_mm = max(maxx - minx, maxy - miny) / TARGET_SIZE_MM
s = 1.0 / px_per_mm
geom = shp_scale(geom, xfact=s, yfact=s, origin=(0, 0))

# 4) Punch a small jump-ring hole near the top of the silhouette.
# Place the hole on the topmost solid column so it actually sits inside the shape.
minx, miny, maxx, maxy = geom.bounds
top_y = maxy
# Find x of the topmost point of the outer boundary
def topmost_x(g):
    if isinstance(g, MultiPolygon):
        # use the polygon whose top is highest
        g = max(g.geoms, key=lambda p: p.bounds[3])
    coords = np.array(g.exterior.coords)
    i = int(np.argmax(coords[:, 1]))
    return float(coords[i, 0])

hx = topmost_x(geom)
hr = HOLE_DIAMETER_MM / 2.0
hy = top_y - HOLE_EDGE_MARGIN_MM - hr  # hole center sits margin+radius below top

hole = Point(hx, hy).buffer(hr, quad_segs=32)
geom_with_hole = geom.difference(hole)
if not geom_with_hole.is_valid:
    geom_with_hole = make_valid(geom_with_hole).buffer(0)
# Ensure the hole is fully inside the shape; if not, nudge down until it is.
tries = 0
while not hole.within(geom) and tries < 20:
    hy -= 0.25
    hole = Point(hx, hy).buffer(hr, quad_segs=32)
    tries += 1
geom = geom.difference(hole)

# 5) Extrude to THICKNESS_MM
def extrude(g):
    if isinstance(g, Polygon):
        return trimesh.creation.extrude_polygon(g, height=THICKNESS_MM)
    return trimesh.util.concatenate(
        [trimesh.creation.extrude_polygon(p, height=THICKNESS_MM) for p in g.geoms]
    )

mesh = extrude(geom)

# 6) Center on origin in XY, sit on z=0
center = (mesh.bounds[0] + mesh.bounds[1]) / 2
mesh.apply_translation([-center[0], -center[1], -mesh.bounds[0, 2]])

mesh.export(DST)

bb = mesh.bounds
print(f"Wrote {DST}")
print(f"Size (mm): X={bb[1,0]-bb[0,0]:.2f}  Y={bb[1,1]-bb[0,1]:.2f}  Z={bb[1,2]-bb[0,2]:.2f}")
print(f"Triangles: {len(mesh.faces)}  Watertight: {mesh.is_watertight}")
