"""
Flask web application for multicolor earring 3MF generation.

Upload a PNG/JPEG → reduce colors → preview → approve → download 3MF.
"""
import base64
import json
import os
import secrets

from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)

from processing import generate_3mf, is_svg, quantized_to_png_bytes, rasterize_svg, reduce_colors

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(32))
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

UPLOAD_DIR = os.path.join(app.root_path, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "svg"}


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _session_path(key: str) -> str:
    """Return a file path scoped to the current session."""
    sid = session.get("sid")
    if not sid:
        sid = secrets.token_hex(16)
        session["sid"] = sid
    d = os.path.join(UPLOAD_DIR, sid)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, key)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Landing page – upload form."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Handle image upload, store it, redirect to preview."""
    file = request.files.get("image")
    if not file or file.filename == "":
        flash("Please select an image file.", "error")
        return redirect(url_for("index"))
    if not _allowed(file.filename):
        flash("Unsupported file type. Please upload a PNG, JPEG, or SVG.", "error")
        return redirect(url_for("index"))

    raw = file.read()
    # Rasterize SVG to PNG so the rest of the pipeline works on bitmap
    if is_svg(raw):
        try:
            raw = rasterize_svg(raw)
        except Exception as exc:
            flash(f"SVG rasterization failed: {exc}", "error")
            return redirect(url_for("index"))
    path = _session_path("original")
    with open(path, "wb") as f:
        f.write(raw)

    n_colors = request.form.get("n_colors", "6", type=str)
    try:
        n_colors = int(n_colors)
        if n_colors < 2:
            n_colors = 2
        if n_colors > 16:
            n_colors = 16
    except ValueError:
        n_colors = 6

    return redirect(url_for("preview", n_colors=n_colors))


@app.route("/preview")
def preview():
    """Show the color-reduced preview for approval."""
    path = _session_path("original")
    if not os.path.exists(path):
        flash("No image uploaded yet.", "error")
        return redirect(url_for("index"))

    n_colors = request.args.get("n_colors", 6, type=int)
    n_colors = max(2, min(16, n_colors))

    with open(path, "rb") as f:
        raw = f.read()

    try:
        quantized, palette = reduce_colors(raw, n_colors)
    except Exception as exc:
        flash(f"Color reduction failed: {exc}", "error")
        return redirect(url_for("index"))

    # Save palette for later
    palette_path = _session_path("palette.json")
    with open(palette_path, "w") as f:
        json.dump(palette, f)

    preview_png = quantized_to_png_bytes(quantized)
    preview_b64 = base64.b64encode(preview_png).decode("ascii")

    return render_template(
        "preview.html",
        preview_b64=preview_b64,
        palette=palette,
        n_colors=n_colors,
    )


@app.route("/generate", methods=["POST"])
def generate():
    """Generate the 3MF and offer it for download."""
    path = _session_path("original")
    palette_path = _session_path("palette.json")
    if not os.path.exists(path) or not os.path.exists(palette_path):
        flash("Session expired. Please upload again.", "error")
        return redirect(url_for("index"))

    with open(path, "rb") as f:
        raw = f.read()
    with open(palette_path, "r") as f:
        palette = [tuple(c) for c in json.load(f)]

    # Gather parameters from form
    target_size_mm = request.form.get("target_size_mm", 39.0, type=float)
    thickness_mm = request.form.get("thickness_mm", 1.0, type=float)
    make_pair = request.form.get("make_pair", "on") == "on"
    nub_width_mm = request.form.get("nub_width_mm", 4.0, type=float)
    nub_height_mm = request.form.get("nub_height_mm", 5.0, type=float)
    hole_diameter_mm = request.form.get("hole_diameter_mm", 1.6, type=float)

    try:
        data = generate_3mf(
            raw,
            palette,
            target_size_mm=target_size_mm,
            thickness_mm=thickness_mm,
            make_pair=make_pair,
            nub_width_mm=nub_width_mm,
            nub_height_mm=nub_height_mm,
            hole_diameter_mm=hole_diameter_mm,
        )
    except Exception as exc:
        flash(f"3MF generation failed: {exc}", "error")
        return redirect(url_for("preview", n_colors=len(palette)))

    out = _session_path("earring.3mf")
    with open(out, "wb") as f:
        f.write(data)

    return redirect(url_for("download_page"))


@app.route("/download")
def download_page():
    """Show download page."""
    out = _session_path("earring.3mf")
    if not os.path.exists(out):
        flash("No generated file found. Please start over.", "error")
        return redirect(url_for("index"))
    return render_template("download.html")


@app.route("/download-file")
def download_file():
    """Serve the generated 3MF file."""
    out = _session_path("earring.3mf")
    if not os.path.exists(out):
        flash("File not found.", "error")
        return redirect(url_for("index"))
    return send_file(
        out,
        mimetype="application/vnd.ms-package.3dmanufacturing-3dmodel+xml",
        as_attachment=True,
        download_name="earring_multicolor.3mf",
    )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
