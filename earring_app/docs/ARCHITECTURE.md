# Architecture

## Overview

Earring Maker is a Flask web application that converts bitmap images (PNG, JPEG, SVG) into multicolor 3MF files for 3D printing earrings. It follows a linear pipeline: upload → color reduce → preview → generate → download.

## Module Responsibilities

### `app.py` — Flask Routes

Thin routing layer. No image processing logic lives here. Responsibilities:

- **`/`** — Render upload form
- **`/upload`** — Accept file, optionally rasterize SVG, optionally remove background, save to session directory, redirect to preview
- **`/preview`** — Run color reduction, generate shape preview, render preview template
- **`/generate`** — Read saved image + palette, call `generate_3mf`, save output
- **`/download`** / **`/download-file`** — Serve the generated 3MF

Session state is file-based: each user gets a directory under `uploads/` keyed by a session token. Files stored: `original` (image bytes), `palette.json`, `filename.txt`, `config.json` (earring dimensions), `earring.3mf`.

### `processing.py` — Core Pipeline

All image processing and 3MF generation. Key function groups:

| Function | Purpose |
|---|---|
| `is_svg`, `rasterize_svg` | SVG detection and rasterization via cairosvg |
| `remove_background` | Flood-fill background removal, outputs RGBA with alpha=0 for background |
| `reduce_colors` | K-means color quantization (OpenCV) |
| `quantized_to_png_bytes` | Encode quantized array to PNG bytes |
| `generate_shape_preview` | Render silhouette+nub on checkered background using Shapely geometry |
| `_build_silhouette` | Extract foreground mask (alpha-based or brightness-based) |
| `_mask_to_polygons` | Convert boolean mask → Shapely Polygon/MultiPolygon via cv2 contours |
| `_add_nub` | Add hanging nub with fillet-smoothed junction and punched hole |
| `_extrude` | Extrude 2D polygon to 3D mesh via trimesh |
| `_mesh_xml`, `_tint` | 3MF XML serialization and vertex coloring |
| `generate_3mf` | Full pipeline: image+palette → 3MF zip bytes |

### Templates

- `base.html` — Layout with skip link, header (logo + theme switcher), footer, flash messages
- `index.html` — Upload form (file, color count, remove-bg checkbox)
- `preview.html` — Shape preview image, palette swatches, color adjustment, 3MF config form
- `download.html` — Download link for generated 3MF

### `static/style.css`

Single stylesheet using CSS custom properties for theming. Three themes: light (default), dark, high-contrast. See [DECISIONS.md](DECISIONS.md) for theme architecture.

### Tests

- `tests/conftest.py` — Shared fixtures (sample PNG, JPEG, SVG, alpha PNG, Flask test client)
- `tests/test_app.py` — Route-level integration tests
- `tests/test_processing.py` — Unit tests for every processing function

## Data Flow

```
Upload (PNG/JPEG/SVG)
  → [rasterize_svg if SVG]
  → [remove_background(tolerance=bg_tolerance, clamped to 1–100) or remove_background_ai(U2NetP via ONNX Runtime) if checked]
  → _cap_image_dimension(raw, 512) keeps longest side ≤ 512 px
  → save raw bytes + config.json (earring dimensions + nub_position + nub_offset_mm) to session dir
  → redirect to /preview

Preview
  → load config.json (target_size_mm, nub_width_mm, nub_height_mm, hole_diameter_mm, thickness_mm, nub_position, nub_offset_mm)
  → reduce_colors(raw, n_colors) with 10 k-means attempts → quantized array + palette
  → generate_shape_preview(quantized_png, palette, raw_bytes=raw, **config)
      caps silhouette to 512 px
      uses _build_silhouette on raw (alpha-aware)
      palette-snaps every pixel to nearest color (same as 3MF pipeline)
      runs black-priority pass with dilation (same as 3MF pipeline)
      determines base color (largest-area palette color)
      uses _add_nub with Shapely fillet and signed nub_offset_mm for smooth, adjustable nub placement
      nub proportions match user-configured dimensions
      rasterizes Shapely geometry back to pixel mask
      composites: silhouette→palette-snapped colors, nub→base color, background→checker
  → render preview.html (config values pre-fill the generate form)

Generate
  → generate_3mf(raw, palette, ...)
      upscale image → build silhouette → quantize → contours → Shapely polygons
      scale to mm → _add_nub with fillet → extrude base + color layers
      center on origin → tint meshes → build 3MF XML
      two parent objects (earring_1, earring_2) referencing shared leaf meshes
      each parent gets its own model_settings.config entry
  → save earring.3mf

Download
  → serve earring.3mf with original filename stem + .3mf extension
```

## Coordinate Systems

**Image coordinates:** origin top-left, y increases downward.

**Shapely coordinates (in `_mask_to_polygons`):** y is flipped (`y = H - row`), so y increases upward. This matches the 3MF/3D convention where the nub is at the "top" (max y).

**3MF coordinates:** millimeters, origin at center of earring after centering translation.
