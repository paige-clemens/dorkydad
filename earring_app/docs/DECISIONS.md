# Design Decisions

This document records architectural and design decisions made during development. AI assistants should read this before making changes to understand *why* things are the way they are.

## D1: No mirror logic — duplicate instead

**Decision:** The 3MF output contains two identical copies of the earring placed side by side, not a mirrored pair.

**Rationale:** Mirror logic was removed because earring images are rarely symmetric and mirroring can invert text or directional art. Two identical copies is simpler and more predictable.

**Implementation:** Two separate parent objects (`earring_1`, `earring_2`) in the 3MF, each referencing the same shared leaf mesh objects. Each parent has its own `<object>` block in `model_settings.config` with full extruder assignments. This is critical — a single parent with two build items causes the slicer to lose color assignments on the second copy.

## D2: Nub junction uses buffer/unbuffer fillet

**Decision:** `_add_nub` unions the nub rectangle+semicircle with the silhouette, then applies `.buffer(fillet_mm).buffer(-fillet_mm)` to smooth the junction.

**Rationale:** Simple union creates sharp concave corners where the nub meets irregular silhouette edges. The dilate-then-erode (Minkowski sum/difference) rounds these corners, producing a smooth, printable transition that works regardless of silhouette shape.

**Default fillet radius:** 1.0 mm. Configurable via `fillet_mm` parameter.

## D3: Preview mirrors the full 3MF pipeline

**Decision:** `generate_shape_preview` replicates the same processing as `generate_3mf`: nearest-color palette snapping, black-priority pass with dilation, base-color detection (largest-area palette color), and Shapely-based `_add_nub` with fillet. The result is a pixel-accurate top-down view of what the printed earring will look like.

**Rationale:** Earlier implementations showed raw quantized pixels or drew the nub with crude OpenCV shapes. The preview now matches the 3MF output exactly — every pixel shows its actual filament slot color, the nub uses the correct base color, and the fillet junction is smooth — so users can approve with confidence before generating.

## D4: Silhouette detection uses raw image, not quantized

**Decision:** `generate_shape_preview` accepts `raw_bytes` (original image with alpha channel) separately from `image_bytes` (quantized/color-reduced image).

**Rationale:** The quantized image is RGB-only (no alpha). When background removal has been applied, the transparent pixels become black (0,0,0) in the quantized PNG. Using the raw image with its alpha channel for `_build_silhouette` correctly identifies the removed background. The quantized image is only used for foreground pixel colors.

## D5: Base color for nub matches 3MF back-layer

**Decision:** The nub in the preview is filled with the palette's base color (the color with the largest pixel area in the silhouette), matching the 3MF generator's back-layer color.

**Rationale:** The nub extends beyond the original image, so there are no source pixels to sample. Using the same base-color logic as `generate_3mf` ensures the preview accurately represents the final print.

## D6: Theme system — CSS custom properties + data attribute

**Decision:** Themes are implemented via CSS custom properties on `:root`, switched by setting `data-theme` on `<html>`. Three modes: light (default), dark, high-contrast.

**Rationale:**
- `prefers-color-scheme: dark` provides automatic dark mode for users with OS-level preference
- Manual toggle overrides via `data-theme` attribute, persisted in `localStorage`
- High-contrast mode uses yellow-on-black with 3px borders, no shadows — designed for low-vision users, separate from OS `forced-colors`
- All color values flow through CSS variables, so no component needs theme-aware logic

## D7: Session-based file storage, not database

**Decision:** User uploads and generated files are stored as plain files in `uploads/<session_id>/`.

**Rationale:** The app processes one image at a time per session. File-based storage is simple, requires no database setup, and works in Docker without additional services. Session IDs are cryptographic random tokens.

## D8: uv for dependency management

**Decision:** Use `uv sync` / `uv run` instead of pip + requirements.txt.

**Rationale:** uv is faster, produces deterministic lockfiles (`uv.lock`), and handles both dependency resolution and virtualenv management. The Dockerfile uses `uv sync --frozen --no-dev` for reproducible production builds.

## D9: Two-file separation (app.py + processing.py)

**Decision:** All image processing and 3MF logic lives in `processing.py`. Flask routing lives in `app.py`.

**Rationale:** Clean separation of concerns. `processing.py` functions are pure (bytes in, bytes out) and independently testable without Flask. `app.py` is a thin orchestration layer handling HTTP, sessions, and file I/O.

## D10: 3MF format targets Bambu Studio / OrcaSlicer

**Decision:** The 3MF includes `Metadata/model_settings.config` with Bambu-style `<part>` elements and extruder assignments.

**Rationale:** This metadata is not part of the 3MF standard but is required by Bambu Studio and OrcaSlicer to correctly assign filament colors to multi-part objects. Without it, the slicer treats everything as a single-color print.

## D11: Earring config set at upload time, not preview time

**Decision:** Size, thickness, nub width, nub height, and hole diameter are configured on the upload form (beginning of workflow) rather than only on the preview/generate step.

**Rationale:** These dimensions directly affect the preview — the nub's proportional size relative to the silhouette changes with `target_size_mm`, and nub shape changes with `nub_width_mm`/`nub_height_mm`/`hole_diameter_mm`. Setting them upfront ensures the preview accurately reflects the final 3MF output. The values are persisted in `config.json` in the session directory, loaded by the preview route, passed to `generate_shape_preview`, and pre-filled in the generate form on the preview page so users can still fine-tune before generation.

## D12: Configurable background-removal aggressiveness

**Decision:** The upload form exposes a `bg_tolerance` range slider (1–100, default 25) that controls the `tolerance` parameter of `remove_background`.

**Rationale:** A fixed tolerance works well for clean solid-color backdrops but fails for noisy photos or backgrounds with slight gradients. The 1–100 range gives fine-grained control: low values only remove colors very close to the corner background color, while high values remove a much wider range of similar colors. This avoids both under-removal (bg artifacts remain) and over-removal (foreground eaten away) on different image types.

## D13: Local AI background-removal test mode

**Decision:** The upload form offers an optional local AI removal mode backed by direct ONNX Runtime inference with the lightweight U2NetP model. The model session is cached in-process after its first use; color-tolerance removal remains the default.

**Rationale:** U2NetP provides a low-resource way to evaluate semantic background removal on a laptop before introducing a GPU service. Its initial inference downloads the model, while later uploads reuse the cached model session. Direct ONNX Runtime avoids incompatible transitive dependencies while retaining the fast, predictable color-tolerance workflow for solid backdrops.

## D14: Nub position and offset

**Decision:** `_add_nub` accepts both `nub_position` (`"left"`, `"center"`, or `"right"`) and `nub_offset_mm` (a signed horizontal offset in millimeters). The nub first attaches at the topmost point (center, default) or at the leftmost/rightmost point among the topmost 2% of the silhouette boundary, then shifts by `nub_offset_mm` (positive = right, negative = left). The final attachment point is clamped so the nub base always overlaps the silhouette.

**Rationale:** Different earring designs hang better from different points. A symmetrical design works best with a centered nub; an asymmetrical one (e.g., a profile silhouette) may look better with a left or right attachment. The offset parameter lets users fine-tune the exact horizontal placement in millimeters rather than being limited to three preset positions.

## D15: Preview speed and interactivity

**Decision:** The upload route caps uploaded images to 512 px on the longest side and the upload/generate buttons show full-screen loading overlays with rotating stage messages. `generate_3mf` also caps its working resolution to 2048 px on the longest side. `reduce_colors` keeps 10 k-means attempts for the best palette quality.

**Rationale:** The slowness was not just the user's local machine — it was caused by processing multi-megapixel images through k-means and Shapely/trimesh operations. A 39 mm earring printed at 0.4 mm nozzle resolution does not benefit from source images larger than ~512 px, and the on-screen preview is displayed at most ~28 rem tall. The resolution cap keeps the upload→preview path fast, the loading overlay gives clear feedback during the unavoidable 3MF generation step, and k-means quality is preserved by keeping 10 attempts.
