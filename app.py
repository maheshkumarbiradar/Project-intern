

"""
app.py - Brain Tumor Classifier (Streamlit UI)
Features:
  - MRI upload & classification
  - Doctor recommendations
  - Download results as PDF
  - Link to MRI Compatibility Checker
Run: python -m streamlit run app.py
"""

import os
import io
import tempfile
import pathlib
import datetime
import numpy as np
import pydicom
import torch
import streamlit as st
from PIL import Image

from predict import predict_image
from doctors import get_tumor_info, recommend_doctors, get_available_cities

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedVision AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif; }
.stApp { background: #0a0f1e; color: #e2e8f0; }
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0d1427 0%, #111827 100%);
    border-right: 1px solid #1e293b;
}
.card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.card-highlight {
    background: linear-gradient(135deg, #0f2027, #1a2a4a);
    border: 1px solid #2563eb44;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.hospital {
    background: #0d1117;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
}
[data-testid="stFileUploader"] {
    border: 2px dashed #1e40af !important;
    border-radius: 12px;
}
/* Primary button */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #7c3aed);
    color: white; border: none; border-radius: 10px;
    font-family: 'Syne', sans-serif; font-weight: 700;
    padding: 0.6rem 2rem; font-size: 1rem; width: 100%;
}
.stButton > button:hover { opacity: 0.88; }
/* Download button override */
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #065f46, #0d9488) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    padding: 0.6rem 2rem !important; font-size: 1rem !important; width: 100% !important;
}
[data-testid="stDownloadButton"] > button:hover { opacity: 0.88 !important; }
/* Link button card */
.link-card {
    background: linear-gradient(135deg, #0f1f3a, #0a1628);
    border: 1px solid #1e40af55;
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    text-align: center;
    cursor: pointer;
    transition: border-color 0.2s;
}
.link-card:hover { border-color: #3b82f6; }
hr { border-color: #1e293b; }
</style>
""", unsafe_allow_html=True)


# ── PDF Generator ─────────────────────────────────────────────────────────────
def dicom_to_pil(file):
    """Read a DICOM upload and return an RGB PIL image compatible with the model."""
    file.seek(0)
    dataset = pydicom.dcmread(file)

    if "PixelData" not in dataset:
        raise ValueError("This DICOM file does not contain pixel data.")

    try:
        pixels = dataset.pixel_array
    except Exception as exc:
        raise ValueError(f"Could not read DICOM pixel data: {exc}") from exc

    # Multi-frame scans arrive as frame stacks. Use the first frame for prediction.
    if pixels.ndim == 4:
        pixels = pixels[0]
    elif pixels.ndim == 3 and not (pixels.shape[-1] in (3, 4)):
        pixels = pixels[0]

    pixels = pixels.astype(np.float32)
    slope = float(getattr(dataset, "RescaleSlope", 1) or 1)
    intercept = float(getattr(dataset, "RescaleIntercept", 0) or 0)
    pixels = pixels * slope + intercept

    min_value = float(np.min(pixels))
    max_value = float(np.max(pixels))
    if max_value > min_value:
        pixels = (pixels - min_value) / (max_value - min_value) * 255.0
    else:
        pixels = np.zeros_like(pixels)

    pixels = np.clip(pixels, 0, 255).astype(np.uint8)

    if pixels.ndim == 2:
        if getattr(dataset, "PhotometricInterpretation", "") == "MONOCHROME1":
            pixels = 255 - pixels
        image = Image.fromarray(pixels, mode="L").convert("RGB")
    else:
        if pixels.shape[-1] == 4:
            pixels = pixels[..., :3]
        image = Image.fromarray(pixels).convert("RGB")

    image.info["dicom_metadata"] = {
        "Patient Name": str(getattr(dataset, "PatientName", "N/A")),
        "Modality": str(getattr(dataset, "Modality", "N/A")),
        "Study Date": str(getattr(dataset, "StudyDate", "N/A")),
    }
    return image


def generate_pdf(result, image, uploaded_name, hospitals, info, city):
    """Generate a PDF report of the scan results."""
    from fpdf import FPDF

    label      = result["label"]
    confidence = result["confidence"]
    all_probs  = result["all_probs"]
    now        = datetime.datetime.now().strftime("%d %B %Y, %I:%M %p")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    font_dir = pathlib.Path(__file__).resolve().parent
    regular_font = font_dir / "NotoSans-Regular.ttf"
    bold_font = font_dir / "NotoSans-Bold.ttf"
    if not regular_font.exists() or not bold_font.exists():
        raise FileNotFoundError(
            "Unicode fonts not found: NotoSans-Regular.ttf and NotoSans-Bold.ttf are required."
        )
    pdf.add_font("Unicode", "", str(regular_font))
    pdf.add_font("Unicode", "B", str(bold_font))

    # ── Header bar ────────────────────────────────────────────────────────────
    pdf.set_fill_color(27, 42, 90)
    pdf.rect(0, 0, 210, 28, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Unicode", "B", 16)
    pdf.set_y(8)
    pdf.cell(0, 10, "MedVision AI - MRI Analysis Report", align="C")
    pdf.set_font("Unicode", "", 9)
    pdf.set_y(18)
    pdf.cell(0, 6, "ResNet18 Transfer Learning | DS & AI ML Track", align="C")

    pdf.set_y(35)

    # ── Report meta ────────────────────────────────────────────────────────────
    pdf.set_text_color(100, 116, 139)
    pdf.set_font("Unicode", "", 9)
    pdf.cell(0, 5, f"Generated: {now}   |   File: {uploaded_name}", align="R")
    pdf.ln(10)

    # ── Prediction result ──────────────────────────────────────────────────────
    color_map = {
        "glioma":     (239, 68, 68),
        "meningioma": (59, 130, 246),
        "pituitary":  (34, 197, 94),
        "notumor":    (148, 163, 184),
    }
    pred_color = color_map.get(label, (148, 163, 184))

    result_box_x = 10
    result_box_y = pdf.get_y()
    result_box_w = 190
    result_box_h = 38
    result_img_size = 28
    result_img_x = result_box_x + result_box_w - result_img_size - 6
    result_img_y = result_box_y + 5

    pdf.set_fill_color(13, 17, 35)
    pdf.set_draw_color(*pred_color)
    pdf.set_line_width(0.8)
    pdf.rect(result_box_x, result_box_y, result_box_w, result_box_h, 'FD')

    try:
        img_copy = image.copy()
        img_copy.thumbnail((300, 300))
        tmp_img = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp_img.close()
        img_copy.save(tmp_img.name)
        pdf.image(tmp_img.name, x=result_img_x, y=result_img_y, w=result_img_size, h=result_img_size)
        os.unlink(tmp_img.name)
    except Exception:
        pass

    pdf.set_y(result_box_y + 6)
    pdf.set_x(result_box_x + 6)
    pdf.set_text_color(148, 163, 184)
    pdf.set_font("Unicode", "", 8)
    pdf.cell(115, 5, "PREDICTED CLASS")

    pdf.set_y(result_box_y + 14)
    pdf.set_x(result_box_x + 6)
    pdf.set_text_color(*pred_color)
    pdf.set_font("Unicode", "B", 20)
    pdf.cell(120, 10, label.upper())

    pdf.set_y(result_box_y + 27)
    pdf.set_x(result_box_x + 6)
    pdf.set_text_color(148, 163, 184)
    pdf.set_font("Unicode", "", 9)
    pdf.cell(120, 5, f"Confidence: {confidence}%")

    pdf.set_y(result_box_y + result_box_h + 6)

    # ── Class probabilities ────────────────────────────────────────────────────
    pdf.set_text_color(27, 42, 90)
    pdf.set_font("Unicode", "B", 11)
    pdf.cell(0, 8, "Class Probabilities", ln=True)

    for cls, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
        c = color_map.get(cls, (148, 163, 184))
        pdf.set_text_color(60, 60, 60)
        pdf.set_font("Unicode", "", 9)
        pdf.cell(45, 6, cls.capitalize())

        # Bar background
        bar_x = pdf.get_x()
        bar_y = pdf.get_y() + 1
        pdf.set_fill_color(230, 235, 245)
        pdf.rect(bar_x, bar_y, 110, 4, 'F')

        # Bar fill
        pdf.set_fill_color(*c)
        fill_w = max(1, 110 * prob / 100)
        pdf.rect(bar_x, bar_y, fill_w, 4, 'F')

        pdf.set_x(bar_x + 115)
        pdf.set_font("Unicode", "B", 9)
        pdf.set_text_color(*c)
        pdf.cell(20, 6, f"{prob}%", align="R")
        pdf.ln(7)

    pdf.ln(4)

    # ── MRI image thumbnail ────────────────────────────────────────────────────
    # ── Medical info ───────────────────────────────────────────────────────────
    pdf.set_draw_color(27, 42, 90)
    pdf.set_fill_color(245, 247, 252)
    pdf.set_line_width(0.3)
    box_y = pdf.get_y()
    pdf.rect(10, box_y, 190, 38, 'FD')
    pdf.set_y(box_y + 4)

    pdf.set_text_color(27, 42, 90)
    pdf.set_font("Unicode", "B", 10)
    pdf.set_x(15)
    pdf.cell(0, 6, "Medical Information", ln=True)

    pdf.set_text_color(60, 60, 60)
    pdf.set_font("Unicode", "B", 9)
    pdf.set_x(15)
    pdf.cell(35, 5, "Specialist:")
    pdf.set_font("Unicode", "", 9)
    pdf.cell(0, 5, info["specialist"], ln=True)

    pdf.set_font("Unicode", "B", 9)
    pdf.set_x(15)
    pdf.cell(35, 5, "Urgency:")
    pdf.set_font("Unicode", "", 9)
    pdf.cell(0, 5, info["urgency"], ln=True)

    pdf.set_font("Unicode", "B", 9)
    pdf.set_x(15)
    pdf.cell(35, 5, "Departments:")
    pdf.set_font("Unicode", "", 9)
    pdf.cell(0, 5, ", ".join(info["departments"]), ln=True)

    pdf.set_font("Unicode", "", 8)
    pdf.set_x(15)
    pdf.set_text_color(100, 116, 139)
    desc = info["description"]
    if len(desc) > 120:
        desc = desc[:120] + "..."
    pdf.cell(0, 5, desc, ln=True)
    pdf.ln(8)

    # ── Hospitals ─────────────────────────────────────────────────────────────
    pdf.set_text_color(27, 42, 90)
    pdf.set_font("Unicode", "B", 11)
    pdf.cell(0, 8, f"Recommended Hospitals — {city}", ln=True)

    for i, h in enumerate(hospitals, 1):
        pdf.set_fill_color(248, 250, 255)
        pdf.set_draw_color(200, 210, 230)
        hy = pdf.get_y()
        pdf.rect(10, hy, 190, 22, 'FD')
        pdf.set_y(hy + 3)

        pdf.set_text_color(27, 42, 90)
        pdf.set_font("Unicode", "B", 10)
        pdf.set_x(14)
        pdf.cell(0, 5, f"{i}. {h['name']}  ★ {h['rating']}", ln=True)

        pdf.set_text_color(80, 80, 80)
        pdf.set_font("Unicode", "", 8)
        pdf.set_x(14)
        pdf.cell(0, 4, h["address"], ln=True)

        pdf.set_x(14)
        pdf.cell(50, 4, h["phone"])
        pdf.set_text_color(27, 90, 173)
        pdf.cell(0, 4, h["website"], ln=True)
        pdf.ln(4)

    # ── Footer ────────────────────────────────────────────────────────────────
    pdf.set_y(-20)
    pdf.set_fill_color(27, 42, 90)
    pdf.rect(0, pdf.get_y(), 210, 20, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Unicode", "", 8)
    pdf.set_y(pdf.get_y() + 4)
    pdf.cell(0, 5,
             "DISCLAIMER: For educational purposes only. Always consult a qualified physician.",
             align="C")

    return bytes(pdf.output())


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 MedVision AI")
    st.markdown("*ResNet18 · Transfer Learning · 4 Classes*")
    st.divider()
    st.markdown("### ⚙️ Settings")
    model_path = st.text_input("Model path", value="best_model.pth")
    city       = st.selectbox("Your city (for doctor recs)", get_available_cities())
    top_n      = st.slider("Hospitals to show", min_value=1, max_value=5, value=3)
    st.divider()
    st.markdown("""
### 📋 Classes
| Label | Tumor Type |
|-------|-----------|
| 🔴 | Glioma |
| 🔵 | Meningioma |
| 🟢 | Pituitary |
| ⚪ | No Tumor |
""")
    st.divider()

    # ── MRI Checker link in sidebar ───────────────────────────────────────────
    st.markdown("### 🔬 Tools")
    st.markdown("""
    <a href="http://localhost:8502" target="_blank" style="text-decoration:none;">
        <div class="link-card">
            <div style="font-size:1.5rem;">🔬</div>
            <div style="font-weight:700; color:#e2e8f0; margin:0.3rem 0;">MRI Compatibility</div>
            <div style="font-size:0.78rem; color:#64748b;">Check image vs training data</div>
        </div>
    </a>
    """, unsafe_allow_html=True)
    st.divider()
    st.caption("⚠️ For educational use only. Always consult a qualified physician.")


# ── Main header ───────────────────────────────────────────────────────────────
col_title, col_tool_btn = st.columns([3, 1])
with col_title:
    st.markdown("<h1 style='font-size:2.6rem; margin-bottom:0.2rem;'>🧠 MedVision AI</h1>",
                unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; font-size:1.05rem;'>Upload an MRI image to classify tumor type and find specialist doctors nearby.</p>",
                unsafe_allow_html=True)
with col_tool_btn:
    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <a href="http://localhost:8502" target="_blank" style="text-decoration:none;">
        <div style="background:linear-gradient(135deg,#0f1f3a,#0a1628);
                    border:1px solid #3b82f655; border-radius:12px;
                    padding:0.85rem 1rem; text-align:center; cursor:pointer;">
            <div style="font-size:1.3rem;">🔬</div>
            <div style="font-size:0.8rem; font-weight:700; color:#3b82f6; margin-top:3px;">
                MRI Compatibility
            </div>
            <div style="font-size:0.7rem; color:#64748b;">Check your image →</div>
        </div>
    </a>
    """, unsafe_allow_html=True)

st.divider()

col_upload, col_result = st.columns([1, 1], gap="large")

# ── Upload column ─────────────────────────────────────────────────────────────
with col_upload:
    st.markdown("### 📤 Upload MRI Image")
    uploaded = st.file_uploader(
        "Drag and drop or click to browse",
        type=["jpg", "jpeg", "png", "dcm"],
        label_visibility="collapsed",
    )

    if uploaded:
        try:
            file_suffix = pathlib.Path(uploaded.name).suffix.lower()
            if file_suffix == ".dcm":
                image = dicom_to_pil(uploaded)
            else:
                uploaded.seek(0)
                image = Image.open(uploaded).convert("RGB")
        except Exception as e:
            st.error(f"Could not read uploaded MRI file: {e}")
            st.stop()

        st.image(image, caption="Uploaded MRI Scan", use_container_width=True)
        dicom_metadata = image.info.get("dicom_metadata")
        if dicom_metadata:
            st.caption(
                " | ".join(f"{key}: {value}" for key, value in dicom_metadata.items())
            )

        if not os.path.exists(model_path):
            st.warning(f"Model file '{model_path}' not found. Train first with: python train.py")
            st.stop()

        analyze_btn = st.button("🔍 Analyze MRI")
    else:
        st.info("👆 Upload a brain MRI image to get started.")
        analyze_btn = False

# ── Result column ─────────────────────────────────────────────────────────────
with col_result:
    st.markdown("### 📊 Analysis Results")

    if uploaded and analyze_btn:
        with st.spinner("Analyzing MRI scan..."):
            tmp_path = str(pathlib.Path(tempfile.gettempdir()) / "uploaded_mri.jpg")
            image.save(tmp_path)
            try:
                result = predict_image(tmp_path, model_path)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        label      = result["label"]
        confidence = result["confidence"]
        all_probs  = result["all_probs"]

        emoji_map   = {"glioma": "🔴", "meningioma": "🔵", "pituitary": "🟢", "notumor": "⚪"}
        label_emoji = emoji_map.get(label, "⚪")

        st.markdown(f"""
        <div class="card-highlight">
            <div style="font-size:0.8rem; color:#64748b; letter-spacing:1px;
                        text-transform:uppercase;">PREDICTION</div>
            <div style="font-size:2.2rem; font-family:'Syne',sans-serif;
                        font-weight:800; margin:0.3rem 0;">
                {label_emoji} {label.capitalize()}
            </div>
            <span style="color:#94a3b8; font-size:0.95rem;">
                Confidence: <b style="color:#e2e8f0;">{confidence}%</b>
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Class Probabilities**")
        color_map = {
            "glioma":     "#ef4444",
            "meningioma": "#3b82f6",
            "pituitary":  "#22c55e",
            "notumor":    "#94a3b8",
        }

        for cls, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
            c1, c2 = st.columns([3, 1])
            color  = color_map.get(cls, "#ffffff")
            with c1:
                st.markdown(f"<small style='color:#94a3b8;'>{cls.capitalize()}</small>",
                            unsafe_allow_html=True)
                st.progress(int(prob))
            with c2:
                st.markdown(
                    f"<div style='text-align:right; padding-top:0.8rem; color:{color};'>"
                    f"<b>{prob}%</b></div>",
                    unsafe_allow_html=True
                )

        st.session_state["result"]   = result
        st.session_state["image"]    = image
        st.session_state["filename"] = uploaded.name

    else:
        st.markdown("""
        <div class="card" style="text-align:center; padding:3rem 1rem; color:#475569;">
            <div style="font-size:3rem;">🩻</div>
            <div style="margin-top:1rem;">Upload and analyze an MRI image<br>to see results here.</div>
        </div>
        """, unsafe_allow_html=True)


# ── Doctor Recommendations + Action Buttons ───────────────────────────────────
if "result" in st.session_state:
    result   = st.session_state["result"]
    image    = st.session_state["image"]
    filename = st.session_state.get("filename", "mri_scan.jpg")
    label    = result["label"]
    info     = get_tumor_info(label)
    hospitals = recommend_doctors(label, city=city, top_n=top_n)

    st.divider()

    # ── ACTION BUTTONS ROW ────────────────────────────────────────────────────
    st.markdown("### ⚡ Quick Actions")
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1], gap="medium")

    # Button 1 — Download PDF
    with btn_col1:
        with st.spinner("Preparing PDF...") if False else st.container():
            try:
                pdf_bytes = generate_pdf(
                    result, image, filename, hospitals, info, city
                )
                pdf_name = f"MedVision_Report_{label}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_bytes,
                    file_name=pdf_name,
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"PDF error: {e}")

    # Button 2 — Open MRI Checker
    with btn_col2:
        st.markdown("""
        <a href="http://localhost:8502" target="_blank" style="text-decoration:none;">
            <div style="background:linear-gradient(135deg,#1e3a5f,#0f2d4a);
                        border:1px solid #3b82f666; border-radius:10px;
                        padding:0.62rem 1rem; text-align:center;
                        font-family:'Syne',sans-serif; font-weight:700;
                        font-size:1rem; color:white; cursor:pointer;
                        display:block; line-height:1.5;">
                🔬 MRI Compatibility Checker
            </div>
        </a>
        """, unsafe_allow_html=True)

    # Button 3 — New Scan
    with btn_col3:
        if st.button("🔄 New Scan", use_container_width=True):
            for key in ["result", "image", "filename"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    st.divider()

    # ── Doctor Recommendations ────────────────────────────────────────────────
    st.markdown("## 🏥 Doctor & Hospital Recommendations")

    col_info, col_hospitals = st.columns([1, 2], gap="large")

    with col_info:
        urgency_level  = info["urgency"].split("—")[0].strip()
        urgency_colors = {"High": "#ef4444", "Moderate": "#f59e0b", "Low": "#22c55e"}
        urgency_color  = urgency_colors.get(urgency_level, "#94a3b8")
        dept_html      = "  ".join(
            f"<code style='font-size:0.75rem;'>{d}</code>"
            for d in info["departments"]
        )

        st.markdown(f"""
        <div class="card">
            <div style="font-size:0.75rem; color:#64748b; text-transform:uppercase; letter-spacing:1px;">
                RECOMMENDED SPECIALIST
            </div>
            <div style="font-size:1.5rem; font-family:'Syne',sans-serif;
                        font-weight:700; margin:0.5rem 0;">
                {info['specialist']}
            </div>
            <div style="margin-top:0.75rem; color:#94a3b8; font-size:0.9rem; line-height:1.6;">
                {info['description']}
            </div>
            <div style="margin-top:1rem; padding:0.5rem 1rem; border-radius:8px;
                        background:#0d1117; border-left:3px solid {urgency_color};">
                <small style="color:#64748b;">URGENCY</small><br>
                <span style="color:{urgency_color}; font-weight:600;">{info['urgency']}</span>
            </div>
            <div style="margin-top:0.75rem;">
                <small style="color:#64748b;">RELEVANT DEPARTMENTS</small><br>
                {dept_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_hospitals:
        if hospitals:
            for h in hospitals:
                st.markdown(f"""
                <div class="hospital">
                    <div style="display:flex; justify-content:space-between; align-items:start;">
                        <div>
                            <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:1.05rem;">
                                {h['name']}
                            </div>
                            <div style="color:#94a3b8; font-size:0.85rem; margin-top:2px;">
                                📍 {h['address']}
                            </div>
                        </div>
                        <div style="text-align:right; color:#fbbf24; font-size:0.85rem;">
                            {h['rating']} ★
                        </div>
                    </div>
                    <div style="margin-top:0.6rem; font-size:0.85rem; color:#64748b;">
                        📞 {h['phone']} &nbsp;·&nbsp;
                        <a href="{h['website']}" target="_blank"
                           style="color:#3b82f6; text-decoration:none;">Visit Website →</a>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info(f"No hospitals found in {city}. Try a different city in the sidebar.")

    st.divider()
    st.caption("⚠️ Medical Disclaimer: This tool is for educational purposes only. "
               "Always consult a qualified physician for medical decisions.")
