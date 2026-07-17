# Coding Standards

Rules and conventions for this project. AI assistants and human contributors should follow these.

## Python

### General

- **Python 3.11+** required (type hints use `X | Y` union syntax)
- No third-party formatting tools enforced; follow the existing style
- Functions in `processing.py` should be pure where possible: bytes in, bytes out
- Avoid adding comments that restate what the code does; prefer docstrings on public functions
- Imports go at the top of the file, grouped: stdlib → third-party → local

### Naming

- Functions: `snake_case`
- Private/internal functions: `_leading_underscore`
- Constants: `UPPER_SNAKE_CASE`
- Classes: `PascalCase` (rarely used — this project is function-oriented)

### Error Handling

- `processing.py` functions raise exceptions (ValueError, etc.) — they don't catch or log
- `app.py` catches exceptions at route boundaries, flashes user-friendly messages, and redirects
- Never silently swallow exceptions

### Type Hints

- Public functions should have type hints on parameters and return values
- Use `list[...]`, `tuple[...]`, `X | None` (not `Optional[X]`)
- NumPy arrays typed as `np.ndarray` (no shape/dtype annotations)

## Testing

- **Framework:** pytest with pytest-cov
- **Coverage target:** ≥80% (currently ~97%)
- **Test location:** `tests/` directory, mirroring source modules (`test_app.py`, `test_processing.py`)
- **Fixtures:** shared fixtures in `tests/conftest.py`
- Test names: `test_<what_it_tests>` — be descriptive
- Tests should not depend on external services or network access
- All test images are generated programmatically in fixtures (no test data files)
- Run tests: `uv run pytest tests/ -v --cov=. --cov-report=term-missing`
- Never delete or weaken existing tests without explicit direction

## HTML / Templates

- **Templating:** Jinja2 (Flask default)
- **Base template:** `base.html` — all pages extend this
- **Accessibility (WCAG 2.1 AA):**
  - Every `<img>` must have a descriptive `alt` attribute
  - Form inputs must have associated `<label>` elements
  - Interactive elements must have visible focus indicators
  - Use semantic HTML (`<main>`, `<nav>`, `<header>`, `<footer>`, `<section>`)
  - Use ARIA attributes where semantic HTML is insufficient
  - Color contrast ≥ 4.5:1 for normal text
  - Support `prefers-reduced-motion`
- Lint warnings in `.html` files about Jinja2 `{{ }}` syntax are false positives from CSS linters — ignore them

## CSS

- **Single file:** `static/style.css`
- **Theming:** All colors, shadows, borders via CSS custom properties on `:root`
- **Theme variants:** override custom properties via `[data-theme="dark"]` and `[data-theme="high-contrast"]` selectors
- **No CSS frameworks** — vanilla CSS with custom properties
- **Responsive:** mobile-first, flex-based layouts, `@media (max-width: 640px)` breakpoint
- Do not use `!important` except in `prefers-reduced-motion` overrides

## 3MF Output

- Each earring gets its own parent `<object>` in the 3MF model
- Both parents reference the same shared leaf mesh `<object>` elements (no geometry duplication)
- `model_settings.config` must have an `<object>` block for **every** parent, each with full `<part>` entries and `extruder` metadata
- Object names include a suffix (`_1`, `_2`) to distinguish copies
- Vertex colors are embedded via trimesh for slicer preview, but extruder assignment in `model_settings.config` is what actually drives filament selection

## Docker

- **Base image:** `python:3.12-slim`
- **System deps:** `libgl1`, `libglib2.0-0`, `libcairo2` (for OpenCV and cairosvg)
- **Build:** `uv sync --frozen --no-dev`
- **Runtime:** gunicorn on port 5000
- **Compose:** single service, port 5000 mapped
- Keep the image minimal — no dev dependencies in production

## Git

- Do not commit `.venv/`, `__pycache__/`, `uploads/`, `.coverage`
- `uv.lock` should be committed for reproducible builds
