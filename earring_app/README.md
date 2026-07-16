# Earring Maker – Multicolor 3MF Generator

A Flask web application that converts PNG/JPEG images into multicolor 3MF files
for 3D printing earrings on multi-filament printers (Bambu X1C, AMS, etc.).

## Features

- Upload PNG or JPEG images
- Automatic color reduction (k-means, 2–16 colors)
- Live preview with approval/rejection workflow
- Configurable earring dimensions, thickness, and hole size
- Earring nub with hole for jump rings automatically added
- Optional mirrored pair generation (left + right ear)
- Multicolor 3MF output compatible with BambuStudio / OrcaSlicer
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

1. **Upload** – Select a PNG/JPEG and choose how many colors (2–16).
2. **Preview** – Review the color-reduced image. Adjust the number of colors and re-process if needed.
3. **Configure** – Set earring size, thickness, nub dimensions, hole diameter, and whether to generate a pair.
4. **Download** – Download the generated `.3mf` file.
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
├── static/
│   └── style.css          # Accessible CSS
└── templates/
    ├── base.html           # Layout with skip-link, header, footer
    ├── index.html           # Upload form
    ├── preview.html         # Color preview + config
    └── download.html        # Download page
```
