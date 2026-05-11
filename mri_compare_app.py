"""
mri_compare_app.py - MRI Image Compatibility Checker (Streamlit UI)
Upload any MRI image and compare it against your training data stats.

Run: python -m streamlit run mri_compare_app.py
"""

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tempfile, pathlib, io

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MRI Compatibility Checker",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Outfit:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }

.stApp {
    background: #060b14;
    color: #e2e8f0;
}

/* Scanline overlay effect */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,255,200,0.01) 2px,
        rgba(0,255,200,0.01) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

[data-testid="stSidebar"] {
    background: #080e1a;
    border-right: 1px solid #0ff2;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #0d1f35, #0a1628);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.75rem;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: var(--bar-color, #00ffc8);
}
.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #4a7fa5;
    margin-bottom: 0.5rem;
}
.metric-values {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}
.metric-train { font-size: 1rem; color: #4a9eff; font-weight: 600; }
.metric-test  { font-size: 1rem; color: #f59e0b; font-weight: 600; }
.metric-diff  { font-size: 1.1rem; font-weight: 700; font-family: 'Space Mono', monospace; }
.progress-bg {
    background: #0d1f35;
    border-radius: 99px;
    height: 6px;
    overflow: hidden;
}
.progress-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.8s ease;
}

/* Score ring */
.score-container {
    text-align: center;
    padding: 2rem 1rem;
    background: linear-gradient(135deg, #0d1f35, #060b14);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    margin-bottom: 1rem;
}
.score-number {
    font-family: 'Space Mono', monospace;
    font-size: 4.5rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.score-label {
    font-size: 0.8rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #4a7fa5;
}
.score-verdict {
    font-size: 1rem;
    font-weight: 600;
    margin-top: 0.75rem;
    padding: 0.4rem 1rem;
    border-radius: 99px;
    display: inline-block;
}

/* Upload zone */
[data-testid="stFileUploader"] {
    border: 2px dashed #0ff3 !important;
    border-radius: 12px;
    background: #0d1f3522 !important;
}

/* Prediction box */
.prediction-box {
    background: #0d1f35;
    border: 1px solid;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-top: 0.5rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    line-height: 1.8;
}

.stButton > button {
    background: linear-gradient(135deg, #00bfa5, #0070c9);
    color: white; border: none; border-radius: 10px;
    font-family: 'Outfit', sans-serif; font-weight: 600;
    font-size: 1rem; width: 100%; padding: 0.65rem;
}
.stButton > button:hover { opacity: 0.85; }

hr { border-color: #1e293b; }
</style>
""", unsafe_allow_html=True)

# ── Training Stats ────────────────────────────────────────────────────────────
DEFAULT_TRAINING_STATS = {
    "brightness": 31.7,
    "contrast":   38.4,
    "width":      512.0,
    "height":     512.0,
    "aspect":     1.0,
    "rgb_pct":    5.0,
}

THRESHOLDS = {
    "brightness": 20.0,
    "contrast":   25.0,
    "width":      30.0,
    "height":     30.0,
    "aspect":     15.0,
    "rgb_pct":    10.0,
}

WEIGHTS = {
    "brightness": 0.35,
    "contrast":   0.25,
    "width":      0.10,
    "height":     0.10,
    "aspect":     0.10,
    "rgb_pct":    0.10,
}

METRIC_LABELS = {
    "brightness": "Brightness",
    "contrast":   "Contrast",
    "width":      "Image Width",
    "height":     "Image Height",
    "aspect":     "Aspect Ratio",
    "rgb_pct":    "RGB vs Grayscale",
}

METRIC_UNITS = {
    "brightness": "pixel avg",
    "contrast":   "std dev",
    "width":      "px",
    "height":     "px",
    "aspect":     "W/H ratio",
    "rgb_pct":    "% RGB",
}

def analyze_image(img: Image.Image):
    mode = img.mode
    arr  = np.array(img.convert('L'))
    w, h = img.size
    return {
        "brightness": float(arr.mean()),
        "contrast":   float(arr.std()),
        "width":      float(w),
        "height":     float(h),
        "aspect":     float(w / h),
        "rgb_pct":    100.0 if mode == 'RGB' else 0.0,
        "mode":       mode,
    }

def pct_diff(test_val, train_val):
    if train_val == 0:
        return 0.0
    return abs(test_val - train_val) / train_val * 100

def compute_score(test_stats, train_stats):
    total = 0.0
    diffs = {}
    for key in WEIGHTS:
        diff = pct_diff(test_stats[key], train_stats[key])
        if key == "rgb_pct":
            diff = abs(test_stats[key] - train_stats[key])
        t     = THRESHOLDS[key]
        score = max(0, 100 - (diff / t) * 50)
        score = min(100, score)
        total += score * WEIGHTS[key]
        diffs[key] = diff
    return total, diffs

def bar_color(diff, threshold):
    if diff <= threshold:
        return "#00ffc8"
    elif diff <= threshold * 2:
        return "#f59e0b"
    return "#ef4444"

def status_emoji(diff, threshold):
    if diff <= threshold:
        return "✅"
    elif diff <= threshold * 2:
        return "⚠️"
    return "🔴"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 MRI Checker")
    st.markdown("*Training Data Reference Stats*")
    st.divider()

    st.markdown("### ⚙️ Training Stats")
    st.caption("Adjust if you rescanned your training folder")

    tr_brightness = st.number_input("Brightness (mean)",    value=31.7, step=0.1)
    tr_contrast   = st.number_input("Contrast (std dev)",   value=38.4, step=0.1)
    tr_width      = st.number_input("Avg Width (px)",       value=512.0, step=1.0)
    tr_height     = st.number_input("Avg Height (px)",      value=512.0, step=1.0)
    tr_rgb        = st.number_input("RGB image % (0-100)",  value=5.0, step=1.0)

    st.divider()
    st.markdown("### 📐 Thresholds")
    st.caption("Max % diff considered acceptable")
    thr_brightness = st.slider("Brightness",   5,  50, 20)
    thr_contrast   = st.slider("Contrast",     5,  50, 25)
    thr_size       = st.slider("Size",        10,  60, 30)
    thr_aspect     = st.slider("Aspect Ratio", 5,  40, 15)
    thr_rgb        = st.slider("RGB Mix",      5,  50, 10)

    THRESHOLDS.update({
        "brightness": float(thr_brightness),
        "contrast":   float(thr_contrast),
        "width":      float(thr_size),
        "height":     float(thr_size),
        "aspect":     float(thr_aspect),
        "rgb_pct":    float(thr_rgb),
    })

    train_stats = {
        "brightness": tr_brightness,
        "contrast":   tr_contrast,
        "width":      tr_width,
        "height":     tr_height,
        "aspect":     tr_width / tr_height if tr_height > 0 else 1.0,
        "rgb_pct":    tr_rgb,
    }

    st.divider()
    st.caption("Training data: Kaggle Brain Tumor MRI\n512×512 px, grayscale, brightness ~31.7")

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='font-family:Space Mono,monospace; font-size:2rem;
           color:#00ffc8; margin-bottom:0.2rem; letter-spacing:2px;'>
    🔬 MRI COMPATIBILITY CHECKER
</h1>
<p style='color:#4a7fa5; font-size:1rem; letter-spacing:1px;'>
    Upload an MRI image → Compare against training data → Get compatibility score
</p>
""", unsafe_allow_html=True)
st.divider()

col_left, col_right = st.columns([1, 1.4], gap="large")

# ── Upload ────────────────────────────────────────────────────────────────────
with col_left:
    st.markdown("### 📤 Upload MRI Image")
    uploaded = st.file_uploader(
        "Drop your MRI image here",
        type=["jpg", "jpeg", "png", "bmp"],
        label_visibility="collapsed"
    )

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption=f"{uploaded.name} ({img.size[0]}×{img.size[1]}px, {img.mode})",
                 use_container_width=True)
        analyze_btn = st.button("⚡ Analyze Compatibility")
    else:
        st.markdown("""
        <div style='border:2px dashed #0ff2; border-radius:12px; padding:3rem 1rem;
                    text-align:center; color:#4a7fa5; background:#0d1f3511;'>
            <div style='font-size:2.5rem;'>🩻</div>
            <div style='margin-top:0.75rem; font-family:Space Mono,monospace;
                        font-size:0.8rem; letter-spacing:1px;'>
                UPLOAD MRI IMAGE<br>TO BEGIN ANALYSIS
            </div>
        </div>
        """, unsafe_allow_html=True)
        analyze_btn = False

# ── Results ───────────────────────────────────────────────────────────────────
with col_right:
    st.markdown("### 📊 Compatibility Results")

    if uploaded and analyze_btn:
        test_stats = analyze_image(img)
        score, diffs = compute_score(test_stats, train_stats)

        # ── Score ────────────────────────────────────────────────────────────
        score_color   = "#00ffc8" if score >= 80 else ("#f59e0b" if score >= 60 else "#ef4444")
        verdict_text  = "GOOD MATCH" if score >= 80 else ("MODERATE" if score >= 60 else "POOR MATCH")
        verdict_bg    = "#00ffc822" if score >= 80 else ("#f59e0b22" if score >= 60 else "#ef444422")

        st.markdown(f"""
        <div class="score-container">
            <div style="font-family:Space Mono,monospace; font-size:0.75rem;
                        letter-spacing:3px; color:#4a7fa5; margin-bottom:0.5rem;">
                COMPATIBILITY SCORE
            </div>
            <div class="score-number" style="color:{score_color};">
                {score:.1f}%
            </div>
            <div style="font-size:0.75rem; color:#4a7fa5; letter-spacing:1px;">
                vs Training Distribution
            </div>
            <div class="score-verdict"
                 style="background:{verdict_bg}; color:{score_color}; border:1px solid {score_color}44;">
                {verdict_text}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Per-metric cards ──────────────────────────────────────────────────
        for key in ["brightness", "contrast", "width", "height", "aspect", "rgb_pct"]:
            diff    = diffs[key]
            thr     = THRESHOLDS[key]
            color   = bar_color(diff, thr)
            emoji   = status_emoji(diff, thr)
            fill_w  = min(100, diff / (thr * 2) * 100)
            tr_val  = train_stats[key]
            ts_val  = test_stats[key]
            unit    = METRIC_UNITS[key]
            label   = METRIC_LABELS[key]

            st.markdown(f"""
            <div class="metric-card" style="--bar-color:{color};">
                <div class="metric-label">{emoji} {label}</div>
                <div class="metric-values">
                    <div>
                        <div style="font-size:0.7rem; color:#4a7fa5; margin-bottom:2px;">TRAINING</div>
                        <div class="metric-train">{tr_val:.1f} <span style="font-size:0.7rem;color:#4a7fa5;">{unit}</span></div>
                    </div>
                    <div>
                        <div style="font-size:0.7rem; color:#4a7fa5; margin-bottom:2px;">YOUR IMAGE</div>
                        <div class="metric-test">{ts_val:.1f} <span style="font-size:0.7rem;color:#4a7fa5;">{unit}</span></div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:0.7rem; color:#4a7fa5; margin-bottom:2px;">DIFFERENCE</div>
                        <div class="metric-diff" style="color:{color};">{diff:.1f}%</div>
                    </div>
                </div>
                <div class="progress-bg">
                    <div class="progress-fill"
                         style="width:{fill_w:.0f}%; background:{color};"></div>
                </div>
                <div style="font-size:0.7rem; color:#4a7fa5; margin-top:4px;">
                    Threshold: ±{thr:.0f}% acceptable
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Store for prediction section
        st.session_state["score"]      = score
        st.session_state["diffs"]      = diffs
        st.session_state["test_stats"] = test_stats

    elif not uploaded:
        st.markdown("""
        <div style='text-align:center; padding:4rem 1rem; color:#4a7fa5;'>
            <div style='font-size:2rem;'>📡</div>
            <div style='font-family:Space Mono,monospace; font-size:0.8rem;
                        letter-spacing:2px; margin-top:1rem;'>
                AWAITING INPUT SIGNAL
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Prediction & Recommendations (full width) ─────────────────────────────────
if "score" in st.session_state:
    score      = st.session_state["score"]
    diffs      = st.session_state["diffs"]
    test_stats = st.session_state["test_stats"]

    st.divider()

    col_pred, col_rec = st.columns([1, 1], gap="large")

    with col_pred:
        st.markdown("### 🎯 Prediction")
        score_color = "#00ffc8" if score >= 80 else ("#f59e0b" if score >= 60 else "#ef4444")

        if score >= 80:
            verdict = "Model should perform WELL"
            detail  = "This image closely matches training distribution. Expected accuracy close to 94%."
            border  = "#00ffc844"
        elif score >= 60:
            verdict = "Model may show MODERATE accuracy drop"
            detail  = "Some domain shift detected. Accuracy may drop 5-15% from baseline. Try --normalize flag."
            border  = "#f59e0b44"
        else:
            verdict = "Model likely to show POOR accuracy"
            detail  = "Significant domain shift detected. Accuracy may drop 20-50%+. Consider fine-tuning."
            border  = "#ef444444"

        st.markdown(f"""
        <div class="prediction-box" style="border-color:{border}; color:{score_color};">
            VERDICT: {verdict}<br><br>
            <span style="color:#94a3b8; font-family:Outfit,sans-serif; font-size:0.9rem;">
                {detail}
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Image info table
        st.markdown("#### 📋 Image Properties")
        img_mode = test_stats.get("mode", "Unknown")
        props = [
            ("Mode", img_mode),
            ("Width", f"{test_stats['width']:.0f} px"),
            ("Height", f"{test_stats['height']:.0f} px"),
            ("Brightness", f"{test_stats['brightness']:.1f}"),
            ("Contrast", f"{test_stats['contrast']:.1f}"),
            ("Aspect Ratio", f"{test_stats['aspect']:.3f}"),
        ]
        for label, value in props:
            st.markdown(
                f"<div style='display:flex; justify-content:space-between; "
                f"padding:0.35rem 0; border-bottom:1px solid #1e293b;'>"
                f"<span style='color:#4a7fa5; font-size:0.9rem;'>{label}</span>"
                f"<span style='font-family:Space Mono,monospace; font-size:0.85rem; "
                f"color:#e2e8f0;'>{value}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

    with col_rec:
        st.markdown("### 💡 Recommendations")

        high_issues = [(METRIC_LABELS[k], diffs[k], THRESHOLDS[k])
                       for k in diffs if diffs[k] > THRESHOLDS[k] * 2]
        warn_issues = [(METRIC_LABELS[k], diffs[k], THRESHOLDS[k])
                       for k in diffs if THRESHOLDS[k] < diffs[k] <= THRESHOLDS[k] * 2]

        if not high_issues and not warn_issues:
            st.success("✅ All metrics within acceptable range. No action needed.")
        else:
            for label, diff, thr in high_issues:
                st.error(f"🔴 **{label}**: {diff:.1f}% difference (threshold: {thr:.0f}%)")
            for label, diff, thr in warn_issues:
                st.warning(f"⚠️ **{label}**: {diff:.1f}% difference (threshold: {thr:.0f}%)")

        st.markdown("#### 🛠️ Actions to Improve Accuracy")

        if score < 60:
            steps = [
                ("1", "Run with brightness normalization",
                 "python evaluate_external.py --data \"<folder>\" --normalize"),
                ("2", "Fine-tune model on samples of this dataset",
                 "Add 50-100 images from this dataset to Training/ and retrain"),
                ("3", "Check image source",
                 "Verify images are from a standard MRI scanner and not screenshots"),
            ]
        elif score < 80:
            steps = [
                ("1", "Try brightness normalization first",
                 "python evaluate_external.py --data \"<folder>\" --normalize"),
                ("2", "Check image mode consistency",
                 "Ensure all images are either RGB or grayscale, not mixed"),
            ]
        else:
            steps = [
                ("1", "No action needed",
                 "Run evaluate_external.py normally without any flags"),
            ]

        for num, title, cmd in steps:
            st.markdown(f"""
            <div style='background:#0d1f35; border:1px solid #1e3a5f;
                        border-radius:10px; padding:0.85rem 1rem; margin-bottom:0.6rem;'>
                <div style='display:flex; gap:0.75rem; align-items:start;'>
                    <div style='background:#00ffc822; color:#00ffc8; font-family:Space Mono,monospace;
                                font-size:0.75rem; padding:0.2rem 0.5rem; border-radius:4px;
                                min-width:1.5rem; text-align:center;'>
                        {num}
                    </div>
                    <div>
                        <div style='font-weight:600; margin-bottom:0.3rem;'>{title}</div>
                        <code style='font-size:0.78rem; color:#4a9eff;
                                     background:#060b14; padding:0.2rem 0.4rem;
                                     border-radius:4px;'>{cmd}</code>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.caption("Reference: Training data = Kaggle Brain Tumor MRI Dataset | 512×512px | Grayscale | Brightness ~31.7")
