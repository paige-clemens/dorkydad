# Earring Maker – Multicolor 3MF Generator

A Flask web application that converts PNG/JPEG images into multicolor 3MF files
for 3D printing earrings on multi-filament printers (Bambu X1C, AMS, etc.).

## Features

- Upload PNG, JPEG, or SVG images
- Optional background removal (flood-fill based)
- Automatic color reduction (k-means, 2–16 colors)
- Live preview with nub, checkered transparency pattern, and smooth fillet contouring
- Configurable earring dimensions, thickness, nub size, and hole diameter
- Earring nub with hole for jump rings, smoothly filleted into irregular silhouettes
- Generates two identical earring copies in the 3MF, placed side by side
- Multicolor 3MF output compatible with BambuStudio / OrcaSlicer
- Dark mode and high contrast mode (with OS preference auto-detection)
- WCAG 2.1 AA accessible UI (skip links, focus rings, ARIA labels, contrast, reduced motion)
- Docker-ready

## Quick Start (local)

Requires [uv](https://docs.astral.sh/uv/).

```bash
cd earring_app
uv sync
uv run python app.py
```

Open http://localhost:5000 in your browser.

## Docker

```bash
cd earring_app
docker compose up --build
```

Open http://localhost:5000 in your browser.

## Workflow

1. **Upload** – Select a PNG/JPEG/SVG, choose color count (2–16), optionally remove background.
2. **Preview** – Review the earring preview showing reduced colors, nub with smooth contouring, and checkered transparency. Adjust color count if needed.
3. **Configure** – Set earring size, thickness, nub dimensions, and hole diameter.
4. **Download** – Download the generated `.3mf` file (named after your original image).
5. **Print** – Open in BambuStudio/OrcaSlicer, assign filaments to each color slot, and print.

## Project Structure

```
earring_app/
├── app.py                 # Flask routes
├── processing.py          # Image processing & 3MF generation
├── pyproject.toml         # Project metadata & dependencies (uv)
├── uv.lock                # Locked dependency versions
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── docs/
│   ├── ARCHITECTURE.md     # System architecture & data flow
│   ├── CODING_STANDARDS.md # Conventions for code, tests, CSS, 3MF
│   └── DECISIONS.md        # Design decisions & rationale
├── static/
│   └── style.css           # Themed, accessible CSS
├── templates/
│   ├── base.html           # Layout with skip-link, header, theme switcher
│   ├── index.html          # Upload form
│   ├── preview.html        # Earring preview + config
│   └── download.html       # Download page
└── tests/
    ├── conftest.py         # Shared fixtures
    ├── test_app.py         # Route integration tests
    └── test_processing.py  # Processing unit tests
```
