import os
import pickle
import random
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import gradio as gr

DISPLAY_IMAGE_SIZE = 256
IMAGE_SIZE   = 32
BATCH_SIZE   = 256
MODEL_DIR    = "/app/models"
DATA_DIR     = "/app/data/celeba-dataset"
ATTR_CSV     = f"{DATA_DIR}/list_attr_celeba.csv"
IMG_DIR      = f"{DATA_DIR}/img_align_celeba/img_align_celeba"
VECTOR_CACHE = "/app/attribute_vectors.pkl"
SAMPLE_FACES_DIR = "/app/sample_faces"

GITHUB_APP = "https://github.com/YuvalKoren07/face-transformation-vae"

ATTRIBUTES = [
    "Eyeglasses", "Smiling", "Young", "Bald", "Pale_Skin",
    "Black_Hair", "Bangs", "Male",
    "Bushy_Eyebrows",
]
ATTR_LABELS = {a: a.replace("_", " ").title() for a in ATTRIBUTES}
ATTR_LABELS["Male"]       = "Gender"
ATTR_LABELS["Black_Hair"] = "Hair Darkness"

print("⏳ Loading encoder / decoder …")
encoder = tf.keras.models.load_model(f"{MODEL_DIR}/encoder")
decoder = tf.keras.models.load_model(f"{MODEL_DIR}/decoder")
print("✅ Models loaded.")


def compute_attribute_vectors():
    if os.path.exists(VECTOR_CACHE):
        print("📦 Loading cached attribute vectors …")
        with open(VECTOR_CACHE, "rb") as f:
            return pickle.load(f)
    print("🔢 Computing attribute vectors (runs once) …")
    attrs_df = pd.read_csv(ATTR_CSV)
    dataset = tf.keras.utils.image_dataset_from_directory(
        IMG_DIR, labels=None, color_mode="rgb",
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE, shuffle=False, interpolation="bilinear",
    )
    all_z = []
    for batch in dataset:
        z_mean, _, _ = encoder.predict(tf.cast(batch, "float32") / 255.0, verbose=0)
        all_z.append(z_mean)
    all_z = np.concatenate(all_z, axis=0)
    n = len(all_z)
    vectors = {}
    for attr in ATTRIBUTES:
        if attr not in attrs_df.columns:
            continue
        labels = attrs_df[attr].values[:n]
        pos_mask, neg_mask = labels == 1, labels == -1
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue
        vectors[attr] = all_z[pos_mask].mean(0) - all_z[neg_mask].mean(0)
        print(f"  ✓  {attr}")
    with open(VECTOR_CACHE, "wb") as f:
        pickle.dump(vectors, f)
    print("✅ Cached to", VECTOR_CACHE)
    return vectors


attribute_vectors = compute_attribute_vectors()
available_attrs   = [a for a in ATTRIBUTES if a in attribute_vectors]
N                 = len(available_attrs)
INIT_CSV          = ",".join(["0.0"] * N)

_cached_z_mean = None


def process_and_encode(image):
    global _cached_z_mean

    if image is None:
        _cached_z_mean = None
        return None, None, None

    pil_img = Image.fromarray(image).convert("RGB")
    small   = pil_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    display = small.resize((DISPLAY_IMAGE_SIZE, DISPLAY_IMAGE_SIZE), Image.NEAREST)

    img_array = np.expand_dims(np.array(small, dtype="float32") / 255.0, axis=0)
    z_mean, _, _ = encoder.predict(img_array, verbose=0)
    _cached_z_mean = z_mean

    recon = (np.clip(decoder.predict(z_mean, verbose=0)[0], 0, 1) * 255).astype(np.uint8)
    recon = np.array(
        Image.fromarray(recon).resize((DISPLAY_IMAGE_SIZE, DISPLAY_IMAGE_SIZE), Image.NEAREST)
    )

    return np.array(display), recon, recon


def load_random_face():
    sample_files = (
        glob.glob(os.path.join(SAMPLE_FACES_DIR, "*.jpg"))
        + glob.glob(os.path.join(SAMPLE_FACES_DIR, "*.jpeg"))
        + glob.glob(os.path.join(SAMPLE_FACES_DIR, "*.png"))
    )
    if not sample_files:
        return None, None, None, None
    chosen = random.choice(sample_files)
    img = np.array(Image.open(chosen).convert("RGB"))
    display, recon, edited = process_and_encode(img)
    return img, display, recon, edited


def apply_sliders(slider_csv):
    global _cached_z_mean
    if _cached_z_mean is None:
        return None
    values = [float(x) for x in slider_csv.split(",")]
    new_z  = _cached_z_mean.copy()
    for attr, strength in zip(available_attrs, values):
        new_z += strength * attribute_vectors[attr]
    edited = decoder.predict(new_z, verbose=0)[0]
    edited = (np.clip(edited, 0, 1) * 255).astype(np.uint8)
    return np.array(
        Image.fromarray(edited).resize((DISPLAY_IMAGE_SIZE, DISPLAY_IMAGE_SIZE), Image.NEAREST)
    )


def reset_sliders():
    return INIT_CSV


def build_slider_html(attrs, labels):
    items = ""
    for i, attr in enumerate(attrs):
        items += f"""
        <div style="padding:10px 6px; border-bottom:1px solid #2a2a3a;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:5px;">
                <span style="color:#ffffff; font-size:1rem;">{labels[attr]}</span>
                <span id="fae-lbl-{i}" style="color:#a78bfa; font-size:.95rem; min-width:34px; text-align:right;">0.0</span>
            </div>
            <input type="range" min="-3" max="3" value="0" step="1"
                   data-idx="{i}"
                   style="width:100%; accent-color:#7c6ff7; cursor:pointer; display:block;">
        </div>"""

    return f"""
    <div id="fae-slider-box" style="
        max-height: 380px;
        overflow-y: scroll;
        overflow-x: hidden;
        background: #16161f;
        border: 1px solid #2a2a3a;
        border-radius: 10px;
        padding: 4px 10px 4px 4px;
        box-sizing: border-box;
        width: 100%;
        scrollbar-width: thin;
        scrollbar-color: #4a4a6a #0f0f14;
    ">{items}</div>"""


JS = f"""
() => {{
    var FAE_N = {N};
    var faeValues = new Array(FAE_N).fill(0.0);
    var faePrevCSV = '{INIT_CSV}';

    function faeGetTextbox() {{
        return document.querySelector('#fae-hidden-box textarea')
            || document.querySelector('#fae-hidden-box input[type="text"]')
            || document.querySelector('#fae-hidden-box input');
    }}

    function faeSendToGradio(csv) {{
        var tb = faeGetTextbox();
        if (!tb) return;
        var proto = tb.tagName === 'TEXTAREA'
            ? window.HTMLTextAreaElement.prototype
            : window.HTMLInputElement.prototype;
        var setter = Object.getOwnPropertyDescriptor(proto, 'value');
        if (setter && setter.set) setter.set.call(tb, csv);
        else tb.value = csv;
        tb.dispatchEvent(new Event('input',  {{bubbles: true}}));
        tb.dispatchEvent(new Event('change', {{bubbles: true}}));
        faePrevCSV = csv;
    }}

    function faeAttachSliders() {{
        var box = document.getElementById('fae-slider-box');
        if (!box) {{ setTimeout(faeAttachSliders, 400); return; }}

        box.addEventListener('input', function(e) {{
            if (e.target.type !== 'range') return;
            var idx = parseInt(e.target.getAttribute('data-idx'));
            var val = parseFloat(e.target.value);
            faeValues[idx] = val;
            var lbl = document.getElementById('fae-lbl-' + idx);
            if (lbl) lbl.innerText = val.toFixed(1);
            faeSendToGradio(faeValues.join(','));
        }});

        setInterval(function() {{
            var tb = faeGetTextbox();
            if (!tb || tb.value === faePrevCSV) return;
            faePrevCSV = tb.value;
            var parts = tb.value.split(',');
            var sliders = box.querySelectorAll('input[type=range]');
            parts.forEach(function(v, i) {{
                var num = parseFloat(v) || 0.0;
                faeValues[i] = num;
                if (sliders[i]) sliders[i].value = num;
                var lbl = document.getElementById('fae-lbl-' + i);
                if (lbl) lbl.innerText = num.toFixed(1);
            }});
        }}, 300);
    }}

    setTimeout(faeAttachSliders, 800);
}}
"""

SECTION_TITLE = (
    "color:#a78bfa; font-size:1rem; font-weight:700; letter-spacing:.1em; "
    "text-transform:uppercase; border-bottom:1px solid #2a2a3a; "
    "padding-bottom:6px; margin-bottom:12px;"
)

CSS = """
body, .gradio-container, .gradio-container > .main, .gradio-container > .main > .wrap {
    background: #0f0f14 !important;
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
}
.gradio-container { padding: 0 16px 24px 16px !important; box-sizing: border-box !important; }
.left-card {
    background: #16161f !important;
    border: 1px solid #2a2a3a !important;
    border-radius: 16px !important;
    padding: 20px !important;
}
.reset-btn {
    background: #1e1e2e !important; border: 1px solid #3a3a55 !important;
    color: #ffffff !important; border-radius: 10px !important;
    margin-top: 10px !important;
}
.random-btn {
    background: #1e2e1e !important; border: 1px solid #3a553a !important;
    color: #ffffff !important; border-radius: 10px !important;
    margin-top: 6px !important;
}
#fae-hidden-box {
    position: absolute !important;
    top: -9999px !important;
    left: -9999px !important;
    width: 1px !important;
    height: 1px !important;
    overflow: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
}
footer { display: none !important; }
"""

with gr.Blocks(css=CSS, js=JS, title="Face Attribute Editor") as demo:

    gr.HTML(f"""
        <div style="text-align:center; padding:24px 0 16px;">
            <h1 style="color:#ffffff; margin:0; font-size:2.2rem; font-weight:700;">
                🎨 Face Attribute Editor
            </h1>
            <p style="color:#a0a0c0; margin-top:8px; font-size:1.25rem;">
                Variational Autoencoder &middot; Latent-space attribute manipulation
            </p>
            <p style="margin-top:8px; font-size:1.1rem;">
                <a href="{GITHUB_APP}" target="_blank"
                   style="color:#a78bfa; text-decoration:none;">
                    📂 View source on GitHub
                </a>
            </p>
        </div>
    """)

    slider_values_box = gr.Textbox(value=INIT_CSV, visible=True, elem_id="fae-hidden-box")

    with gr.Row(equal_height=False):

        # ── LEFT COLUMN ──────────────────────────────────────────
        with gr.Column(scale=1, elem_classes="left-card"):

            gr.HTML(f'<p style="{SECTION_TITLE}">🎚 Attributes</p>')
            gr.HTML(f"""
                <p style="color:#cccccc; font-size:1rem; margin:0 0 10px 0;">
                    Drag the scrollbar on the right to see all.
                    Negative values reverse an attribute.
                </p>
            """)
            gr.HTML(build_slider_html(available_attrs, ATTR_LABELS))

            reset_btn  = gr.Button("↺ Reset All Sliders", elem_classes="reset-btn")

            gr.HTML(f'<p style="{SECTION_TITLE} margin-top:20px;">📁 Upload Photo</p>')
            input_image = gr.Image(label="Upload a face photo", type="numpy")
            random_btn  = gr.Button("🎲 Random Face", elem_classes="random-btn")

        # ── RIGHT COLUMN ─────────────────────────────────────────
        with gr.Column(scale=3):

            with gr.Row(equal_height=True):
                with gr.Column():
                    gr.HTML(f'<p style="{SECTION_TITLE}">🔽 32×32 Downscaled</p>')
                    input_display = gr.Image(
                        label="32x32 Downscaled",
                        height=DISPLAY_IMAGE_SIZE,
                        interactive=False
                    )

                with gr.Column():
                    gr.HTML(f'<p style="{SECTION_TITLE}">🔍 Decoded / Reconstructed</p>')
                    original_out = gr.Image(
                        label="Decoded / Reconstructed",
                        height=DISPLAY_IMAGE_SIZE,
                        interactive=False
                    )

                with gr.Column():
                    gr.HTML(f'<p style="{SECTION_TITLE}">✨ Edited</p>')
                    edited_out = gr.Image(
                        label="After edits",
                        height=DISPLAY_IMAGE_SIZE,
                        interactive=False
                    )

            gr.HTML(f"""
                <p style="{SECTION_TITLE} margin-top:20px;">ℹ️ How it works</p>
                <div style="background:#1a1a28; border-left:4px solid #a78bfa;
                            border-radius:8px; padding:22px 24px; margin-bottom:20px;">
                    <p style="color:#ffffff; font-size:1.15rem; line-height:2.4; margin:0;">
                        <b style="color:#c4affe;">0. Train</b> &mdash;
                            I defined and trained a Variational Autoencoder on the
                            <a href="http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html" target="_blank"
                               style="color:#a78bfa;">CelebA faces dataset</a>
                            (200k images) &mdash;
                            <a href="https://github.com/YuvalKoren07/face-transformation-vae/blob/main/Vae_Training_final.ipynb" target="_blank"
                               style="color:#a78bfa;">view training notebook</a>.<br>
                        <b style="color:#c4affe;">1. Encode</b> &mdash;
                            Your photo is resized to 32&times;32 and mapped to a
                            400-D point in the VAE latent space.<br>
                        <b style="color:#c4affe;">2. Shift</b> &mdash;
                            Each slider nudges that point toward or away from
                            a learned attribute direction.<br>
                        <b style="color:#c4affe;">3. Decode</b> &mdash;
                            The decoder renders the shifted latent point back
                            into a face image.
                    </p>
                </div>

                <p style="{SECTION_TITLE}">🗒 Tips</p>
                <div style="background:#1a1a28; border-left:4px solid #6c63ff;
                            border-radius:8px; padding:22px 24px;">
                    <p style="color:#ffffff; font-size:1.15rem; line-height:2.4; margin:0;">
                        &#x2022; The <b style="color:#c4affe;">Decoded / Reconstructed</b>
                            image shows how well the VAE encoded your photo.<br>
                        &#x2022; Outputs look <b style="color:#c4affe;">somewhat blurry</b>
                            &mdash; the model was trained on 32&times;32 images.<br>
                        &#x2022; Use <b style="color:#c4affe;">&plusmn;2&ndash;3</b> for
                            dramatic changes,
                            <b style="color:#c4affe;">&plusmn;1</b> for subtle ones.<br>
                        &#x2022; Combining multiple sliders gives the most realistic results.<br>
                        &#x2022; Works best with a well-lit, front-facing photo where
                            <b style="color:#c4affe;">the face takes up ~50% of the image</b>.
                    </p>
                </div>
            """)

    # ── Event wiring ─────────────────────────────────────────────
    input_image.change(
        fn=process_and_encode,
        inputs=[input_image],
        outputs=[input_display, original_out, edited_out],
    )
    random_btn.click(
        fn=load_random_face,
        outputs=[input_image, input_display, original_out, edited_out],
    )
    slider_values_box.change(
        fn=apply_sliders,
        inputs=[slider_values_box],
        outputs=[edited_out],
    )
    reset_btn.click(
        fn=reset_sliders,
        outputs=[slider_values_box],
    )


if __name__ == "__main__":
    print("Running on local URL:  http://localhost:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True, show_api=False, quiet=True)
