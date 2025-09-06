#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clothing Overlay Prototype — Single-file Streamlit App (app.py)

Features
- Two modes (radio at top):
  A) Upload & Apply  — upload a person photo + clothing image, then Apply
  B) Preset Examples — pick from 3 demo photos and 3 demo clothing items
- Right panel shows original photo, clothing, and result side-by-side (or stacked on mobile)
- Mock mode using Pillow for local overlay (alpha composite + simple warp/rescale)
- Hook function call_gemini_overlay(...) with a clear TODO region to integrate the real Gemini call
- Privacy banner and README collapsible block
- Auto-resize images to max 1024 px long edge
- Download result button
- No user images saved to disk

Run
  pip install -r requirements.txt
  # macOS/Linux
  export GEMINI_API_KEY="..."
  # Windows PowerShell
  setx GEMINI_API_KEY "..."
  streamlit run app.py
"""

from __future__ import annotations
import io
import os
import base64
import textwrap
from typing import Dict, Tuple, Optional

import streamlit as st
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageDraw, ImageChops
import requests

# =============================================================================
# Configuration & Constants
# =============================================================================

st.set_page_config(
    page_title="Clothing Overlay Prototype",
    layout="wide",
)

MAX_EDGE = 1024  # maximum size in px for the longest edge
APP_TITLE = "Clothing Overlay Prototype"

# PLACE API KEY HERE
# Optionally put your default API key here for local testing (NOT recommended for prod).
# You can also use environment var GEMINI_API_KEY. If both exist, env var takes precedence.
DEFAULT_API_KEY: Optional[str] = None  # e.g., "AIza..."  # DO NOT COMMIT REAL KEYS

PRIVACY_NOTE = (
    "Images uploaded in free AI Studio may be used to improve models. "
    "Do not upload sensitive content."
)

# =============================================================================
# Utilities
# =============================================================================

def read_env_api_key() -> Optional[str]:
    """Return the Gemini API key from (in order):
    1) st.secrets["GEMINI_API_KEY"] (Streamlit Cloud Secrets)
    2) environment variable GEMINI_API_KEY
    3) DEFAULT_API_KEY (hardcoded fallback; not recommended)
    """
    key = None
    try:
        # Streamlit Cloud secrets (or local .streamlit/secrets.toml)
        key = st.secrets.get("GEMINI_API_KEY")  # type: ignore[attr-defined]
    except Exception:
        # st.secrets may not exist in some contexts
        key = None
    return key or os.environ.get("GEMINI_API_KEY") or DEFAULT_API_KEY


def api_key_source() -> str:
    """Return a human-readable source of the API key for UI display."""
    # Try secrets first
    try:
        if st.secrets.get("GEMINI_API_KEY"):
            return "Secrets"
    except Exception:
        pass
    if os.environ.get("GEMINI_API_KEY"):
        return "Env"
    if DEFAULT_API_KEY:
        return "Code Default"
    return "None"


def resize_image_max(img: Image.Image, max_edge: int = MAX_EDGE) -> Image.Image:
    """Resize image so that the longest edge <= max_edge, preserving aspect ratio."""
    if img is None:
        return img
    w, h = img.size
    longest = max(w, h)
    if longest <= max_edge:
        return img
    scale = max_edge / float(longest)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size, resample=Image.LANCZOS)


def ensure_rgba(img: Image.Image) -> Image.Image:
    """Convert an image to RGBA mode."""
    if img.mode != "RGBA":
        return img.convert("RGBA")
    return img


def image_to_bytes_png(img: Image.Image) -> bytes:
    """Encode PIL Image to PNG bytes."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def bytes_to_image_safe(data: bytes) -> Image.Image:
    """Decode bytes into a PIL Image with safety defaults."""
    img = Image.open(io.BytesIO(data))
    img.load()
    return img

# =============================================================================
# Demo Asset Generation (in-memory, no disk IO)
# =============================================================================

@st.cache_data(show_spinner=False)
def generate_demo_photo(index: int, size: Tuple[int, int] = (900, 1200)) -> bytes:
    """Generate a simple demo person photo as PNG bytes.
    This uses abstract shapes to avoid any real person data.
    """
    w, h = size
    # Soft background
    bg = Image.new("RGB", size, (240, 240, 245))
    bg = ImageEnhance.Brightness(bg).enhance(1.05)

    # Simple mannequin-like figure (head + torso)
    canvas = bg.convert("RGBA")
    layer = Image.new("RGBA", size, (0, 0, 0, 0))

    # Head
    head_radius = int(min(w, h) * 0.08)
    head = Image.new("RGBA", (head_radius * 2, head_radius * 2), (0, 0, 0, 0))
    head_alpha = Image.new("L", head.size, 0)
    d = ImageDraw.Draw(head_alpha)
    d.ellipse([(0, 0), (head.size[0]-1, head.size[1]-1)], fill=255)
    head.putalpha(head_alpha)

    # Torso rectangle
    torso_w = int(w * 0.36)
    torso_h = int(h * 0.38)
    torso = Image.new("RGBA", (torso_w, torso_h), (220, 205, 200, 255))

    # Position head and torso
    head_x = w // 2 - head.size[0] // 2
    head_y = int(h * 0.12)
    torso_x = w // 2 - torso_w // 2
    torso_y = head_y + head.size[1] + int(h * 0.03)

    layer.alpha_composite(head, (head_x, head_y))
    layer.alpha_composite(torso, (torso_x, torso_y))

    combined = Image.alpha_composite(canvas, layer)

    # Soft blur for a photographic look
    combined = combined.filter(ImageFilter.GaussianBlur(radius=0.5))

    return image_to_bytes_png(combined.convert("RGB"))


@st.cache_data(show_spinner=False)
def generate_demo_clothing(index: int, size: Tuple[int, int] = (800, 800)) -> bytes:
    """Generate a simple clothing PNG with transparency (e.g., shirt-like shape)."""
    w, h = size
    base = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(base)

    colors = [
        (60, 120, 255, 220),   # blue
        (255, 80, 100, 220),   # red/pink
        (40, 200, 150, 220),   # teal/green
    ]
    col = colors[(index - 1) % len(colors)]

    # Shirt silhouette
    body_w, body_h = int(w * 0.65), int(h * 0.55)
    body_x = w // 2 - body_w // 2
    body_y = int(h * 0.25)

    # Body
    draw.rounded_rectangle(
        [body_x, body_y, body_x + body_w, body_y + body_h],
        radius=int(body_w * 0.08), fill=col
    )
    # Neck opening
    neck_r = int(body_w * 0.14)
    draw.ellipse(
        [w // 2 - neck_r, body_y - neck_r // 2, w // 2 + neck_r, body_y + neck_r // 2],
        fill=(0, 0, 0, 0)
    )
    # Sleeves
    sleeve_w = int(body_w * 0.42)
    sleeve_h = int(body_h * 0.35)
    draw.pieslice(
        [body_x - sleeve_w // 2, body_y + int(sleeve_h * 0.2), body_x + int(sleeve_w * 0.7), body_y + sleeve_h],
        270, 360, fill=col
    )
    draw.pieslice(
        [body_x + body_w - int(sleeve_w * 0.7), body_y + int(sleeve_h * 0.2), body_x + body_w + sleeve_w // 2, body_y + sleeve_h],
        180, 270, fill=col
    )

    # Subtle pattern
    pattern = Image.new("RGBA", size, (0, 0, 0, 0))
    pd = ImageDraw.Draw(pattern)
    for i in range(12):
        alpha = 30 if i % 2 == 0 else 20
        pd.arc([body_x + 10, body_y + 10, body_x + body_w - 10, body_y + body_h - 10],
               start=10 * i, end=10 * i + 145, fill=(255, 255, 255, alpha), width=2)
    base = Image.alpha_composite(base, pattern)

    return image_to_bytes_png(base)


@st.cache_data(show_spinner=False)
def get_demo_assets() -> Dict[str, Dict[str, bytes]]:
    """Return demo assets as bytes without writing to disk.

    Returns a dict with keys 'photos' and 'clothes', each mapping names to PNG bytes.
    """
    photos = {
        "Demo Photo 1": generate_demo_photo(1),
        "Demo Photo 2": generate_demo_photo(2),
        "Demo Photo 3": generate_demo_photo(3),
    }
    clothes = {
        "Shirt A (Blue)": generate_demo_clothing(1),
        "Shirt B (Red)": generate_demo_clothing(2),
        "Shirt C (Teal)": generate_demo_clothing(3),
    }
    return {"photos": photos, "clothes": clothes}

# =============================================================================
# Mock overlay (simple alpha composite + mild warp)
# =============================================================================

import math

def simple_perspective_warp(img: Image.Image, strength: float = 0.08) -> Image.Image:
    """Apply a mild perspective-like warp to the clothing image for realism.

    This constructs a simple quad transform that narrows the top slightly.
    """
    img = ensure_rgba(img)
    w, h = img.size
    dx = int(w * strength)
    # Source quad corners
    src = [
        (0, 0), (w, 0), (w, h), (0, h)
    ]
    # Destination quad (narrower top)
    dst = [
        (0 + dx, 0), (w - dx, 0), (w, h), (0, h)
    ]
    return img.transform((w, h), Image.QUAD, data=sum(dst, ()), resample=Image.BICUBIC)


def _estimate_bg_color(img: Image.Image) -> Tuple[int, int, int]:
    """Estimate background color from image corners (assumes near-uniform BG)."""
    rgb = img.convert("RGB")
    w, h = rgb.size
    samples = []
    k = max(1, min(w, h) // 20)
    corners = [
        (0, 0), (w - k, 0), (0, h - k), (w - k, h - k)
    ]
    for (cx, cy) in corners:
        region = rgb.crop((cx, cy, min(cx + k, w), min(cy + k, h)))
        colors = list(region.getdata())
        if colors:
            r = sum(c[0] for c in colors) // len(colors)
            g = sum(c[1] for c in colors) // len(colors)
            b = sum(c[2] for c in colors) // len(colors)
            samples.append((r, g, b))
    if not samples:
        return (255, 255, 255)
    r = sum(s[0] for s in samples) // len(samples)
    g = sum(s[1] for s in samples) // len(samples)
    b = sum(s[2] for s in samples) // len(samples)
    return (r, g, b)


def _remove_background_auto(img: Image.Image, threshold: int = 35) -> Image.Image:
    """Remove a near-uniform background by chroma distance from corner-estimated color.

    Returns an RGBA image with alpha where foreground kept. Also crops to tight bbox.
    """
    img = ensure_rgba(img)
    # If already has transparency, keep as-is but still try to auto-trim.
    bg = _estimate_bg_color(img)
    rgb = img.convert("RGB")
    w, h = rgb.size
    # Build alpha mask based on distance from bg color
    mask = Image.new("L", (w, h), 0)
    pix_rgb = rgb.load()
    pix_m = mask.load()
    thr2 = threshold * threshold
    for y in range(h):
        for x in range(w):
            pr, pg, pb = pix_rgb[x, y]
            dr = pr - bg[0]
            dg = pg - bg[1]
            db = pb - bg[2]
            dist2 = dr * dr + dg * dg + db * db
            if dist2 > thr2:
                pix_m[x, y] = 255
            else:
                pix_m[x, y] = 0
    # Smooth and expand a bit to avoid halos
    mask = mask.filter(ImageFilter.MaxFilter(3))
    mask = mask.filter(ImageFilter.MedianFilter(3))
    mask = mask.filter(ImageFilter.GaussianBlur(1.2))

    out = img.copy()
    out.putalpha(mask)
    # Crop to content
    bbox = mask.getbbox()
    if bbox:
        out = out.crop(bbox)
    return out


def mock_overlay(
    photo_rgba: Image.Image,
    clothing_rgba: Image.Image,
    *,
    remove_bg: bool = True,
    width_pct: int = 60,
    y_pct: int = 32,
    rotation_deg: float = -3.0,
    perspective: float = 0.08,
) -> Image.Image:
    """Produce a plausible local overlay using Pillow only.

    Steps:
    - Convert to RGBA
    - Optionally remove background from clothing (auto)
    - Resize clothing to width_pct% width of photo
    - Mild perspective warp + rotation
    - Paste clothing centered around upper torso region (y_pct)
    """
    base = ensure_rgba(photo_rgba)
    cloth = ensure_rgba(clothing_rgba)

    if remove_bg:
        cloth = _remove_background_auto(cloth)

    pw, ph = base.size
    target_w = int(pw * (width_pct / 100.0))
    scale = target_w / float(cloth.width)
    target_h = max(1, int(cloth.height * scale))
    cloth = cloth.resize((target_w, target_h), resample=Image.LANCZOS)

    # Mild warp and rotation
    cloth = simple_perspective_warp(cloth, strength=float(perspective))
    cloth = cloth.rotate(angle=float(rotation_deg), resample=Image.BICUBIC, expand=True)

    # Place around upper-torso region
    x = pw // 2 - cloth.width // 2
    y = int(ph * (y_pct / 100.0))

    out = base.copy()
    out.alpha_composite(cloth, dest=(x, y))
    return out

# =============================================================================
# Gemini hook function
# =============================================================================

def call_gemini_overlay(
    photo_bytes: bytes,
    clothing_bytes: bytes,
    prompt_text: str,
    mock: bool = False,
    api_key: Optional[str] = None,
) -> bytes:
    """Return PNG bytes for the overlay result.

    If mock=True or api_key is missing, uses local Pillow mock.

    Args:
        photo_bytes: PNG/JPEG bytes for the person photo
        clothing_bytes: PNG/JPEG bytes for the clothing image (prefer RGBA with transparency)
        prompt_text: Full prompt to send to the model
        mock: Force mock mode
        api_key: Gemini API key (if None, env var will be checked inside caller)

    Returns:
        PNG bytes of the composited result
    """
    # If we are asked to mock or no usable API key is provided, use Pillow-based mock.
    if mock or not api_key:
        photo_img = bytes_to_image_safe(photo_bytes)
        clothing_img = bytes_to_image_safe(clothing_bytes)
        photo_img = resize_image_max(photo_img, MAX_EDGE)
        clothing_img = resize_image_max(clothing_img, MAX_EDGE)
        # Read overlay options from session if available
        opts = st.session_state.get("overlay_options", {}) if hasattr(st, "session_state") else {}
        result = mock_overlay(
            photo_img,
            clothing_img,
            remove_bg=bool(opts.get("remove_bg", True)),
            width_pct=int(opts.get("width_pct", 60)),
            y_pct=int(opts.get("y_pct", 32)),
            rotation_deg=float(opts.get("rotation_deg", -3.0)),
            perspective=float(opts.get("perspective", 0.08)),
        )
        return image_to_bytes_png(result)

    # Real call path using Gemini Generative Language API over HTTP
    try:
        model_name = st.session_state.get("model_name", "gemini-2.5-flash-image")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        headers = {"Content-Type": "application/json"}
        params = {"key": api_key}
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt_text},
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": base64.b64encode(photo_bytes).decode("utf-8"),
                            }
                        },
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": base64.b64encode(clothing_bytes).decode("utf-8"),
                            }
                        },
                    ],
                }
            ],
            # Ask for image/png output; supported on image-capable models
            "generationConfig": {"response_mime_type": "image/png"},
        }

        resp = requests.post(url, headers=headers, params=params, json=payload, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text[:512]}")
        data = resp.json()
        # Expect inline data in first candidate
        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError("No candidates returned from Gemini.")
        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            raise RuntimeError("No parts in Gemini response.")
        # Find first inline_data
        for p in parts:
            if "inline_data" in p:
                b64png = p["inline_data"].get("data")
                if b64png:
                    return base64.b64decode(b64png)
        # Some models might return a data_uri field
        for p in parts:
            if "data" in p and isinstance(p["data"], str):
                return base64.b64decode(p["data"])  # best effort
        raise RuntimeError("Gemini response did not include image data.")
    except Exception as ex:
        # Fallback to mock on any error so the app stays usable
        st.error(f"Gemini call failed; falling back to mock. Details: {ex}")
        photo_img = bytes_to_image_safe(photo_bytes)
        clothing_img = bytes_to_image_safe(clothing_bytes)
        opts = st.session_state.get("overlay_options", {}) if hasattr(st, "session_state") else {}
        result = mock_overlay(
            photo_img,
            clothing_img,
            remove_bg=bool(opts.get("remove_bg", True)),
            width_pct=int(opts.get("width_pct", 60)),
            y_pct=int(opts.get("y_pct", 32)),
            rotation_deg=float(opts.get("rotation_deg", -3.0)),
            perspective=float(opts.get("perspective", 0.08)),
        )
        return image_to_bytes_png(result)

    # For now, fall back to mock if we ever reach here.
    photo_img = bytes_to_image_safe(photo_bytes)
    clothing_img = bytes_to_image_safe(clothing_bytes)
    opts = st.session_state.get("overlay_options", {}) if hasattr(st, "session_state") else {}
    result = mock_overlay(
        photo_img,
        clothing_img,
        remove_bg=bool(opts.get("remove_bg", True)),
        width_pct=int(opts.get("width_pct", 60)),
        y_pct=int(opts.get("y_pct", 32)),
        rotation_deg=float(opts.get("rotation_deg", -3.0)),
        perspective=float(opts.get("perspective", 0.08)),
    )
    return image_to_bytes_png(result)

# =============================================================================
# Prompt Template
# =============================================================================

DEFAULT_PROMPT_TEMPLATE = textwrap.dedent(
    """
    System:
    You are a precise visual stylist assistant. You receive two images:
    1) USER_PHOTO: A person front-facing or near-frontal.
    2) GARMENT_IMAGE: A top-wear garment with transparency around the silhouette.

    Task:
    - Fit the garment realistically onto the person in USER_PHOTO.
    - Maintain garment proportions, neck opening alignment, and plausible shoulder placement.
    - Preserve skin, hair, and background. Do not erase the head.
    - Honor transparency from GARMENT_IMAGE (holes, neck, etc).
    - Return a single PNG image with the overlaid result.

    Constraints:
    - Do not crop unless necessary; preserve the entire person photo framing.
    - If the person is partially occluded, make a best effort placement.
    - Subtle perspective/warp is allowed to match shoulder slope.

    Output:
    - A single image/png binary of the final composite.

    References:
    - USER_PHOTO is provided as file: user_photo.png
    - GARMENT_IMAGE is provided as file: garment.png

    Notes:
    - Prefer top-center placement at the upper torso region.
    - If alignment is ambiguous, prioritize neck alignment, then shoulder width.

    End of instructions.
    """
).strip()

# =============================================================================
# UI Helpers
# =============================================================================

def show_header_and_privacy():
    st.title(APP_TITLE)
    st.caption(PRIVACY_NOTE)


def sidebar_controls() -> Tuple[bool, bool, str]:
    """Render toggles and prompt editor.

    Returns:
        use_real (bool): True if real Gemini should be used
        mock_mode (bool): True if mock mode explicitly enabled
        prompt (str): The prompt text
    """
    st.sidebar.markdown("### Settings")

    api_key_present = bool(read_env_api_key())
    use_real = st.sidebar.toggle(
        "Use real Gemini",
        value=api_key_present,
        help="Enable to call Gemini if API key is set in env or Streamlit Secrets.",
        disabled=not api_key_present,
    )
    mock_mode = st.sidebar.checkbox(
        "Mock mode",
        value=not use_real,
        help="Use local Pillow overlay. Works without an API key.",
    )

    with st.sidebar.expander("Prompt template — editable", expanded=False):
        prompt = st.text_area(
            label="Prompt sent to model",
            value=DEFAULT_PROMPT_TEMPLATE,
            height=260,
        )

        st.caption(
            "This prompt is sent along with two files: user_photo.png and garment.png.\n"
            "Tune alignment instructions, perspective hints, or safety notes here."
        )

    st.sidebar.caption(f"API key source: {api_key_source()}")

    # Model selector (used when real Gemini is enabled)
    model = st.sidebar.selectbox(
        "Gemini model",
        options=[
            "gemini-2.5-flash-image",
            "gemini-2.0-flash-exp",
            "gemini-1.5-flash",
        ],
        index=0,
        help="Model used for real calls. Must support image IO.",
    )

    with st.sidebar.expander("Try-on controls (mock)", expanded=False):
        remove_bg = st.checkbox("Auto remove garment background", value=True,
                                help="Tries to remove a near-uniform background from the clothing image. Works best if background is solid.")
        width_pct = st.slider("Garment width (% of photo width)", min_value=30, max_value=95, value=60, step=1)
        y_pct = st.slider("Vertical position (upper torso %)", min_value=20, max_value=55, value=32, step=1)
        rotation_deg = st.slider("Rotation (degrees)", min_value=-15.0, max_value=15.0, value=-3.0, step=0.5)
        perspective = st.slider("Perspective strength", min_value=0.0, max_value=0.3, value=0.08, step=0.01)

        st.session_state["overlay_options"] = {
            "remove_bg": remove_bg,
            "width_pct": width_pct,
            "y_pct": y_pct,
            "rotation_deg": rotation_deg,
            "perspective": perspective,
        }

    st.session_state["model_name"] = model

    with st.sidebar.expander("README / How to run", expanded=False):
        st.markdown(
            textwrap.dedent(
                """
                Run locally:
                - pip install -r requirements.txt
                - Set API key:
                  - macOS/Linux: export GEMINI_API_KEY="..."
                  - Windows PowerShell: setx GEMINI_API_KEY "..."
                - streamlit run app.py

                Integration notes:
                - See call_gemini_overlay() and the section labeled:
                  - # PLACE API KEY HERE
                  - # TODO: IMPLEMENT ACTUAL GEMINI CALL
                  Replace the mock path with real Gemini SDK calls and return PNG bytes.
                
                Streamlit Cloud (via GitHub):
                - Deploy the repo on Streamlit Community Cloud
                - In the app's Settings → Secrets, add GEMINI_API_KEY = your_key
                - The app will automatically detect st.secrets and enable "Use real Gemini"
                """
            )
        )

    return use_real, mock_mode, prompt


def crop_center_square(img: Image.Image) -> Image.Image:
    """Utility for thumbnail previews in a square aspect."""
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    return img.crop((left, top, left + s, top + s))


# =============================================================================
# Main App Views
# =============================================================================

def view_upload_and_apply(use_real: bool, mock_mode: bool, prompt_text: str):
    left, right = st.columns([1, 1.2])

    with left:
        st.subheader("Option A — Upload & Apply")
        user_photo_file = st.file_uploader("Upload your photo", type=["png", "jpg", "jpeg"]) 
        clothing_file = st.file_uploader("Upload clothing image (prefer PNG with transparency)", type=["png", "jpg", "jpeg"]) 
        apply_clicked = st.button("Apply", type="primary")

        if apply_clicked:
            if not user_photo_file or not clothing_file:
                st.error("Please upload both a photo and a clothing image.")
            else:
                try:
                    with st.spinner("Applying overlay..."):
                        photo_bytes = user_photo_file.read()
                        clothing_bytes = clothing_file.read()
                        # Auto-resize safety
                        photo_img = resize_image_max(bytes_to_image_safe(photo_bytes), MAX_EDGE)
                        clothing_img = resize_image_max(bytes_to_image_safe(clothing_bytes), MAX_EDGE)
                        photo_bytes = image_to_bytes_png(photo_img)
                        clothing_bytes = image_to_bytes_png(clothing_img)

                        api_key = read_env_api_key() if use_real and not mock_mode else None
                        if use_real and not mock_mode and not api_key:
                            st.error("GEMINI_API_KEY not found. Set the environment variable or enable Mock mode.")
                        else:
                            result_bytes = call_gemini_overlay(
                                photo_bytes=photo_bytes,
                                clothing_bytes=clothing_bytes,
                                prompt_text=prompt_text,
                                mock=(mock_mode or not api_key),
                                api_key=api_key,
                            )
                            st.session_state["last_result"] = result_bytes
                            st.session_state["last_photo"] = photo_bytes
                            st.session_state["last_clothing"] = clothing_bytes
                except Exception as e:
                    st.error(f"Overlay failed: {e}")

    with right:
        st.subheader("Preview")
        p_bytes = st.session_state.get("last_photo")
        c_bytes = st.session_state.get("last_clothing")
        r_bytes = st.session_state.get("last_result")

        if any([p_bytes, c_bytes, r_bytes]):
            c1, c2, c3 = st.columns(3)
            if p_bytes:
                with c1:
                    st.caption("Original Photo")
                    st.image(p_bytes, caption="User Photo", use_container_width=True)
            if c_bytes:
                with c2:
                    st.caption("Clothing Image")
                    st.image(c_bytes, caption="Garment", use_container_width=True)
            if r_bytes:
                with c3:
                    st.caption("Result")
                    st.image(r_bytes, caption="Overlay Result", use_container_width=True)
                    st.download_button(
                        "Download Result",
                        data=r_bytes,
                        file_name="overlay_result.png",
                        mime="image/png",
                    )
        else:
            st.info("Upload images and click Apply to see results.")


def view_preset_examples(use_real: bool, mock_mode: bool, prompt_text: str):
    left, right = st.columns([1, 1.2])
    assets = get_demo_assets()

    with left:
        st.subheader("Option B — Preset Examples")
        photo_name = st.selectbox("Choose a demo photo", list(assets["photos"].keys()), index=0)
        clothing_name = st.selectbox("Choose a demo clothing", list(assets["clothes"].keys()), index=0)
        apply_demo = st.button("Apply to Demo", type="primary")

        if apply_demo:
            try:
                with st.spinner("Applying overlay to demo..."):
                    photo_bytes = assets["photos"][photo_name]
                    clothing_bytes = assets["clothes"][clothing_name]

                    # Ensure sizes reasonable
                    photo_img = resize_image_max(bytes_to_image_safe(photo_bytes), MAX_EDGE)
                    clothing_img = resize_image_max(bytes_to_image_safe(clothing_bytes), MAX_EDGE)
                    photo_bytes = image_to_bytes_png(photo_img)
                    clothing_bytes = image_to_bytes_png(clothing_img)

                    api_key = read_env_api_key() if use_real and not mock_mode else None
                    if use_real and not mock_mode and not api_key:
                        st.error("GEMINI_API_KEY not found. Set the environment variable or enable Mock mode.")
                    else:
                        result_bytes = call_gemini_overlay(
                            photo_bytes=photo_bytes,
                            clothing_bytes=clothing_bytes,
                            prompt_text=prompt_text,
                            mock=(mock_mode or not api_key),
                            api_key=api_key,
                        )
                        st.session_state["last_result"] = result_bytes
                        st.session_state["last_photo"] = photo_bytes
                        st.session_state["last_clothing"] = clothing_bytes
            except Exception as e:
                st.error(f"Overlay failed: {e}")

    with right:
        st.subheader("Preview")
        p_bytes = st.session_state.get("last_photo")
        c_bytes = st.session_state.get("last_clothing")
        r_bytes = st.session_state.get("last_result")

        if any([p_bytes, c_bytes, r_bytes]):
            c1, c2, c3 = st.columns(3)
            if p_bytes:
                with c1:
                    st.caption("Original Photo")
                    st.image(p_bytes, caption="Demo Photo", use_container_width=True)
            if c_bytes:
                with c2:
                    st.caption("Clothing Image")
                    st.image(c_bytes, caption="Garment", use_container_width=True)
            if r_bytes:
                with c3:
                    st.caption("Result")
                    st.image(r_bytes, caption="Overlay Result", use_container_width=True)
                    st.download_button(
                        "Download Result",
                        data=r_bytes,
                        file_name="overlay_result.png",
                        mime="image/png",
                    )
        else:
            st.info("Pick a photo and clothing, then click Apply to Demo.")

# =============================================================================
# App Entry
# =============================================================================

def main():
    show_header_and_privacy()

    # Top-level mode selection
    mode = st.radio(
        "Choose mode",
        ["Upload & Apply", "Preset Examples"],
        horizontal=True,
    )

    # Sidebar settings & prompt
    use_real, mock_mode, prompt_text = sidebar_controls()

    # Persist last results across reruns
    st.session_state.setdefault("last_photo", None)
    st.session_state.setdefault("last_clothing", None)
    st.session_state.setdefault("last_result", None)

    if mode == "Upload & Apply":
        view_upload_and_apply(use_real, mock_mode, prompt_text)
    else:
        view_preset_examples(use_real, mock_mode, prompt_text)


if __name__ == "__main__":
    main()
