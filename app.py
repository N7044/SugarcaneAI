"""
app.py - SugarcaneAI: Explainable AI & RAG Sugarcane Gap Detection System
Streamlit Application Entrypoint.
"""

import os
import io
import time
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from detector import SugarcaneDetector
from explainability import OcclusionSensitivityExplainer, SHAPExtensionPoint
from rag_engine import SugarcaneRAGEngine
from report_generator import SugarcaneReportGenerator, image_to_base64

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="SugarcaneAI | Gap Detection & Agronomic Decision Support",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Agricultural Professional CSS
st.markdown("""
<style>
    /* Theme Palette */
    :root {
        --primary: #1b4332;
        --secondary: #2d6a4f;
        --accent: #52b788;
        --bg-light: #f8faf8;
        --card-border: #d8e2dc;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 100%);
        color: white;
        padding: 24px;
        border-radius: 10px;
        margin-bottom: 25px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.1rem;
        font-weight: 700;
        color: #ffffff;
    }
    .main-header p {
        margin: 6px 0 0 0;
        font-size: 0.98rem;
        color: #d8f3dc;
    }
    
    .stMetric {
        background: #f8faf8;
        border: 1px solid #d8e2dc;
        border-radius: 8px;
        padding: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    
    .metric-container {
        display: flex;
        gap: 15px;
        margin-bottom: 20px;
    }
    
    .badge-status-ok {
        background-color: #d8f3dc;
        color: #1b4332;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.82rem;
        display: inline-block;
    }
    
    .badge-status-warn {
        background-color: #fff3cd;
        color: #856404;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.82rem;
        display: inline-block;
    }

    .badge-status-err {
        background-color: #f8d7da;
        color: #721c24;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.82rem;
        display: inline-block;
    }

    .callout-box {
        background-color: #f1f8f3;
        border-left: 4px solid #2d6a4f;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 14px 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "session_history" not in st.session_state:
    st.session_state.session_history = []
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "current_image_name" not in st.session_state:
    st.session_state.current_image_name = "sample_field.jpg"
if "detection_results" not in st.session_state:
    st.session_state.detection_results = None
if "xai_result" not in st.session_state:
    st.session_state.xai_result = None
if "rag_history" not in st.session_state:
    st.session_state.rag_history = []
if "last_rag_result" not in st.session_state:
    st.session_state.last_rag_result = None
if "last_rag_query" not in st.session_state:
    st.session_state.last_rag_query = None

# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
st.sidebar.markdown("## ⚙️ Model Configuration")

# Model Path Selection
root_model_path = os.path.join(os.path.dirname(__file__), "best.pt")
models_dir_model_path = os.path.join(os.path.dirname(__file__), "models", "best.pt")
default_model_path = root_model_path if os.path.exists(root_model_path) else models_dir_model_path
model_option = st.sidebar.radio(
    "Weights Source",
    ["Default Path (models/best.pt)", "Custom Local Path", "Upload .pt Weights"],
    index=0
)

selected_model_path = default_model_path
if model_option == "Custom Local Path":
    selected_model_path = st.sidebar.text_input("Enter absolute path to .pt file:", value=default_model_path)
elif model_option == "Upload .pt Weights":
    uploaded_weights = st.sidebar.file_uploader("Upload trained YOLOv8 weights (.pt)", type=["pt"])
    if uploaded_weights is not None:
        target_upload_path = os.path.join(os.path.dirname(__file__), "models", "uploaded_best.pt")
        with open(target_upload_path, "wb") as f:
            f.write(uploaded_weights.getbuffer())
        selected_model_path = target_upload_path
        st.sidebar.success(f"Loaded uploaded weights: {uploaded_weights.name}")

# Model Caching Loader
@st.cache_resource(show_spinner=False)
def get_detector(model_path: str) -> SugarcaneDetector:
    return SugarcaneDetector(model_path)

detector = get_detector(selected_model_path)

# Display Model Status
if detector.is_loaded:
    st.sidebar.markdown(f'<div class="badge-status-ok">● Model Loaded: {os.path.basename(selected_model_path)}</div>', unsafe_allow_html=True)
    st.sidebar.markdown(f"**Detected Classes ({len(detector.class_names)}):**")
    for cid, cname in detector.class_names.items():
        st.sidebar.markdown(f"- Class `{cid}`: **{cname}**")
else:
    st.sidebar.markdown('<div class="badge-status-warn">▲ Model Missing / Unloaded</div>', unsafe_allow_html=True)
    st.sidebar.caption(f"Status: {detector.load_error}")
    st.sidebar.info("👉 Place your trained `best.pt` in `models/` or select 'Upload .pt Weights' above.")

st.sidebar.markdown("---")
st.sidebar.markdown("## 🎛️ Inference Settings")
conf_thresh = st.sidebar.slider("Confidence Threshold", min_value=0.05, max_value=0.95, value=0.25, step=0.05)
iou_thresh = st.sidebar.slider("NMS IoU Threshold", min_value=0.10, max_value=0.90, value=0.45, step=0.05)
img_size = st.sidebar.select_slider("Inference Image Size", options=[320, 480, 640, 800, 960], value=640)

st.sidebar.markdown("---")
st.sidebar.markdown("## 🤖 RAG & LLM Assistant")
configured_key = os.getenv("GEMINI_API_KEY", "")
gemini_key_input = st.sidebar.text_input(
    "Gemini API Key (Optional)", 
    value=configured_key,
    type="password", 
    help="If omitted, RAG runs in retrieval-only mode using local TF-IDF."
)
if gemini_key_input:
    os.environ["GEMINI_API_KEY"] = gemini_key_input
    st.sidebar.markdown('<div class="badge-status-ok">● Gemini API Active</div>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<div class="badge-status-warn">⚪ Retrieval-Only Mode (Local)</div>', unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_rag_engine() -> SugarcaneRAGEngine:
    kb_path = os.path.join(os.path.dirname(__file__), "knowledge_base")
    return SugarcaneRAGEngine(kb_path)

rag_engine = get_rag_engine()
st.sidebar.caption(f"Knowledge Base: **{len(rag_engine.chunks)} indexed chunks**")

# ---------------------------------------------------------
# App Header
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🌱 SugarcaneAI</h1>
    <p>Explainable AI & Retrieval-Augmented Generation System for Sugarcane Gap Detection & Agronomic Decision Support</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Top Navigation Tabs
# ---------------------------------------------------------
tabs = st.tabs([
    "1. Overview",
    "2. Gap Detection",
    "3. Explainability Lab (XAI)",
    "4. Agriculture RAG Assistant",
    "5. Reports & History",
    "6. About the Model"
])

# =========================================================
# TAB 1: OVERVIEW
# =========================================================
with tabs[0]:
    st.subheader("System Architecture & Operational Pipeline")
    
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        **SugarcaneAI** addresses early-stage crop establishment failure in commercial sugarcane plantations by pairing
        deep computer vision with empirical explainability and evidence-grounded agronomic decision support.
        
        ### Key Functional Modules:
        1. **Feature A — Computer Vision Detection:** Powered by your custom-trained YOLOv8 model, identifying crop discontinuities and stand gaps with bounding-box metrics.
        2. **Feature B — Explainable AI (XAI):** Real perturbation-based *Occlusion Sensitivity Analysis*, revealing which visual image regions drove the model's confidence.
        3. **Feature C — Retrieval-Augmented Generation (RAG):** Fast local TF-IDF & cosine vector retrieval over agronomic knowledge bases, with strict LLM grounding and citation traceability.
        4. **Feature D — Analytics Dashboard:** Real-time metrics, class distributions, and confidence analytics computed strictly from inference results.
        5. **Feature E — Agronomic Reporting:** Print-ready, downloadable HTML reports distinguishing visual detections from ground-scouting requirements.
        """)
        
        st.info("💡 **Getting Started:** Move to the **Gap Detection** tab to load an aerial field image or select the built-in sample image to run inference.")
    
    with col2:
        st.markdown("### Model & System Status")
        if detector.is_loaded:
            st.success(f"**Detector Ready:** `{os.path.basename(selected_model_path)}`")
            st.write(f"- **Classes Count:** {len(detector.class_names)}")
            st.write(f"- **Classes Names:** {', '.join([f'{k}: {v}' for k, v in detector.class_names.items()])}")
            st.write(f"- **Default Size:** {img_size}px")
        else:
            st.warning("**Model Weights Awaiting Placement**")
            st.write(f"Default path checked: `{selected_model_path}`")
            st.caption("Upload or place your trained `best.pt` file to enable detection.")

        st.markdown("### Agronomic Knowledge Engine")
        st.write(f"- **Indexed Corpus Chunks:** {len(rag_engine.chunks)}")
        st.write(f"- **Vector Space:** TF-IDF with Unigram/Bigram representation")
        st.write(f"- **LLM Provider:** Google Gemini API (Configured: {'✅ Active' if (gemini_key_input or os.getenv('GEMINI_API_KEY')) else '⚪ Retrieval-Only Mode'})")

# =========================================================
# TAB 2: GAP DETECTION
# =========================================================
with tabs[1]:
    st.subheader("Automated Sugarcane Stand Gap Detection")

    img_source = st.radio("Image Input Mode:", ["Use Built-in Sample Field Image", "Upload Image(s) (JPG, JPEG, PNG)"], horizontal=True)

    target_pil_image = None
    target_img_name = "sample_field.jpg"

    if img_source == "Use Built-in Sample Field Image":
        sample_path = os.path.join(os.path.dirname(__file__), "sample_data", "sample_field.jpg")
        if os.path.exists(sample_path):
            target_pil_image = Image.open(sample_path)
            target_img_name = "sample_field.jpg"
        else:
            st.error("Sample image file not found in `sample_data/`.")
    else:
        uploaded_files = st.file_uploader("Upload Drone/Field Photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if uploaded_files:
            # Let user pick which uploaded image to analyze if multiple
            if len(uploaded_files) > 1:
                selected_file_name = st.selectbox("Select file from batch:", [f.name for f in uploaded_files])
                selected_file = next(f for f in uploaded_files if f.name == selected_file_name)
            else:
                selected_file = uploaded_files[0]
            target_pil_image = Image.open(selected_file)
            target_img_name = selected_file.name

    if target_pil_image is not None:
        st.session_state.current_image = target_pil_image
        st.session_state.current_image_name = target_img_name

        btn_run = st.button("🚀 Run YOLOv8 Gap Detection", type="primary")

        if btn_run:
            if not detector.is_loaded:
                st.error("Cannot run inference: YOLOv8 model weights are not loaded. Please provide your `best.pt` file.")
            else:
                with st.spinner("Executing YOLOv8 inference..."):
                    results = detector.predict(
                        image=target_pil_image,
                        conf_threshold=conf_thresh,
                        iou_threshold=iou_thresh,
                        imgsz=img_size
                    )
                    st.session_state.detection_results = results
                    st.session_state.xai_result = None  # Reset XAI for new image

                    # Save to session history
                    st.session_state.session_history.append({
                        "image_name": target_img_name,
                        "timestamp": time.strftime("%H:%M:%S"),
                        "total_detections": results["stats"]["total"],
                        "mean_confidence": results["stats"]["mean_conf"],
                        "class_counts": results["class_counts"]
                    })

    # Render Results if available
    res = st.session_state.detection_results
    if res is not None:
        st.markdown("---")
        st.markdown("### Detection Results & Field Analytics")

        # Metric cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Detections", res["stats"]["total"])
        m2.metric("Mean Confidence", f"{res['stats']['mean_conf']:.3f}" if res['stats']['total'] > 0 else "0.000")
        m3.metric("Min Confidence", f"{res['stats']['min_conf']:.3f}" if res['stats']['total'] > 0 else "0.000")
        m4.metric("Max Confidence", f"{res['stats']['max_conf']:.3f}" if res['stats']['total'] > 0 else "0.000")

        # Side by side visual comparison
        c_orig, c_annot = st.columns(2)
        with c_orig:
            st.markdown("**Original Field Image**")
            st.image(st.session_state.current_image, use_container_width=True)
        with c_annot:
            st.markdown(f"**Annotated Detections (Threshold: {conf_thresh:.2f})**")
            st.image(res["annotated_image"], use_container_width=True)

        # Download annotated image
        buf_img = io.BytesIO()
        res["annotated_image"].save(buf_img, format="PNG")
        st.download_button(
            label="📥 Download Annotated Image (PNG)",
            data=buf_img.getvalue(),
            file_name=f"annotated_{st.session_state.current_image_name}.png",
            mime="image/png"
        )

        # Interactive Plotly Charts
        if res["stats"]["total"] > 0:
            st.markdown("### Quantitative Distribution Charts")
            ch_col1, ch_col2 = st.columns(2)

            with ch_col1:
                class_df = pd.DataFrame(list(res["class_counts"].items()), columns=["Class", "Count"])
                fig_bar = px.bar(
                    class_df, x="Class", y="Count",
                    color="Class",
                    title="Detection Count by Class",
                    color_discrete_sequence=px.colors.qualitative.Dark24
                )
                fig_bar.update_layout(showlegend=False, height=320, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_bar, use_container_width=True)

            with ch_col2:
                conf_values = [d["confidence"] for d in res["detections"]]
                fig_hist = px.histogram(
                    x=conf_values, nbins=10,
                    title="Confidence Score Distribution",
                    labels={"x": "Confidence Score", "y": "Frequency"},
                    color_discrete_sequence=["#2d6a4f"]
                )
                fig_hist.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_hist, use_container_width=True)

            # Detection Table
            st.markdown("### Detection Details & Spatial Bounding Coordinates")
            det_df = detector.to_dataframe(res["detections"])
            st.dataframe(det_df, use_container_width=True)

            # CSV Export
            csv_data = det_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Detections (CSV)",
                data=csv_data,
                file_name=f"detections_{st.session_state.current_image_name}.csv",
                mime="text/csv"
            )
        else:
            st.info("ℹ️ No detections met the current confidence threshold. Consider adjusting the confidence slider in the sidebar if faint gaps are present.")

        st.caption("⚠️ **Methodological Boundary:** Detected boxes represent 2D image pixel intervals. They do not represent calibrated ground meters or an exact count of dead setts without physical ground scouting.")

# =========================================================
# TAB 3: EXPLAINABILITY LAB (XAI)
# =========================================================
with tabs[2]:
    st.subheader("Explainable AI: Occlusion Sensitivity Analysis")
    st.markdown("""
    Unlike classification where Grad-CAM is standard, object detection requires targeted explainability.
    **Occlusion Sensitivity** tests whether masking small spatial patches diminishes the model's confidence
    for a chosen target detection, constructing a mathematically verifiable attribution heatmap.
    """)

    res = st.session_state.detection_results

    if res is None or res["stats"]["total"] == 0:
        st.warning("Please run gap detection first on an image with at least one detection.")
    else:
        det_options = [
            f"Detection #{d['detection_id']} - {d['class_name']} (Conf: {d['confidence']:.2f})"
            for d in res["detections"]
        ]
        selected_det_str = st.selectbox("Select Target Detection to Explain:", det_options)
        selected_idx = det_options.index(selected_det_str)
        target_det = res["detections"][selected_idx]

        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            patch_sz = st.select_slider("Occlusion Patch Size (px)", options=[24, 32, 48, 64], value=48)
        with col_p2:
            stride_step = st.select_slider("Sliding Stride Step (px)", options=[16, 24, 32, 48], value=24,
                                           help="Smaller stride yields higher heatmap fidelity but requires more model inferences.")
        with col_p3:
            mask_type = st.selectbox("Mask Color", ["Neutral Gray (128,128,128)", "Black (0,0,0)", "Soil Brown (145,115,80)"])
            color_map = {
                "Neutral Gray (128,128,128)": (128, 128, 128),
                "Black (0,0,0)": (0, 0, 0),
                "Soil Brown (145,115,80)": (145, 115, 80)
            }
            selected_color = color_map[mask_type]

        btn_explain = st.button("🔬 Compute Occlusion Sensitivity Heatmap", type="primary")

        if btn_explain:
            explainer = OcclusionSensitivityExplainer(detector)
            prog_bar = st.progress(0.0)
            status_text = st.empty()

            def update_progress(frac: float, msg: str):
                prog_bar.progress(frac)
                status_text.text(msg)

            with st.spinner("Running perturbation experiments across image grid..."):
                xai_out = explainer.explain_detection(
                    image=st.session_state.current_image,
                    target_detection=target_det,
                    patch_size=patch_sz,
                    stride=stride_step,
                    mask_color=selected_color,
                    progress_callback=update_progress
                )
                st.session_state.xai_result = xai_out
                status_text.text("Perturbation analysis complete!")

        # Render XAI Results
        xai_res = st.session_state.xai_result
        if xai_res is not None:
            st.markdown("---")
            st.markdown("### Occlusion Sensitivity Findings")

            xm1, xm2, xm3, xm4 = st.columns(4)
            xm1.metric("Original Confidence", f"{xai_res['original_score']:.3f}")
            xm2.metric("Min Conf Under Mask", f"{xai_res['min_perturbed_score']:.3f}")
            xm3.metric("Max Sensitivity Drop", f"{xai_res['max_sensitivity_drop']:.3f}")
            xm4.metric("Inferences Evaluated", f"{xai_res['total_evaluations']} ({xai_res['runtime_sec']}s)")

            xc1, xc2 = st.columns(2)
            with xc1:
                st.markdown("**Original Field Image with Target BBox**")
                # Crop or highlight target box
                st.image(st.session_state.current_image, use_container_width=True)
            with xc2:
                st.markdown("**Occlusion Sensitivity Heatmap Overlay**")
                st.image(xai_res["heatmap_overlay"], use_container_width=True)

            st.markdown(f'<div class="callout-box"><strong>Attribution Summary:</strong> {xai_res["method_explanation"]}</div>', unsafe_allow_html=True)

            # Download XAI Heatmap
            buf_xai = io.BytesIO()
            xai_res["heatmap_overlay"].save(buf_xai, format="PNG")
            st.download_button(
                label="📥 Download Sensitivity Heatmap (PNG)",
                data=buf_xai.getvalue(),
                file_name=f"xai_heatmap_target_{target_det['detection_id']}.png",
                mime="image/png"
            )

        # Documented SHAP Extension Section
        with st.expander("ℹ️ Architecture Note: SHAP Integration & Theoretical Differences"):
            shap_info = SHAPExtensionPoint.get_extension_documentation()
            st.markdown(f"""
            - **Status:** `{shap_info['status']}`
            - **Recommended Algorithm:** `{shap_info['recommended_algorithm']}`
            - **Scalar Target Function:** `{shap_info['scalar_objective']}`
            - **Computational Complexity:** `{shap_info['computational_profile']}`
            - **Technical Explanation:** Standard SHAP was designed for single scalar vector predictions. Object detectors
              produce hundreds of candidate anchor boxes simultaneously. Occlusion sensitivity provides an exact, empirically
              traceable scalar objective (class-specific IoU-matched confidence drop) without heuristic tensor slicing.
            - **Extension Spec:** {shap_info['interface_spec']}
            """)

# =========================================================
# TAB 4: AGRICULTURE RAG ASSISTANT
# =========================================================
with tabs[3]:
    st.subheader("Agronomic Knowledge & Decision Support Assistant")
    st.markdown("""
    Ask questions regarding sugarcane germination, gap management, replanting thresholds, and visual remote sensing protocols.
    Answers are retrieved from the curated agronomic knowledge base with strict evidence grounding.
    """)

    # Quick prompt selection
    sample_queries = [
        "What causes poor germination and gap formation in sugarcane?",
        "What are the agronomic thresholds for replanting or gap filling?",
        "What are the critical limitations of drone and vision-based gap detection?",
        "When is the best time after planting to perform gap filling?",
        "How should field scouts verify detected gaps before replanting?"
    ]
    selected_sample = st.selectbox("Select a standard agronomic inquiry or enter your own below:", ["-- Custom Query --"] + sample_queries)

    default_q = "" if selected_sample == "-- Custom Query --" else selected_sample
    user_query = st.text_input("Agronomist Inquiry:", value=default_q, placeholder="e.g., What are the best practices for gap filling with nursery seedlings?")

    col_rag_btn, col_rag_k = st.columns([2, 1])
    with col_rag_k:
        top_k_select = st.slider("Top Chunks Retrieved", min_value=2, max_value=5, value=3)
    with col_rag_btn:
        st.write("")
        btn_ask_rag = st.button("🔍 Query Knowledge Assistant", type="primary")

    if btn_ask_rag and user_query.strip():
        with st.spinner("Searching agronomic vector space & synthesizing grounded response..."):
            rag_output = rag_engine.query_assistant(
                query=user_query,
                api_key=gemini_key_input,
                top_k=top_k_select
            )
            st.session_state.last_rag_result = rag_output
            st.session_state.last_rag_query = user_query
            st.session_state.rag_history.append({
                "query": user_query,
                "answer": rag_output["answer"],
                "citations": rag_output["citations"],
                "confidence": rag_output["retrieval_confidence"]
            })

    # Display RAG Output
    rag_res = st.session_state.last_rag_result
    if rag_res is not None and st.session_state.last_rag_query:
        st.markdown("---")
        st.markdown(f"### Response for: *'{st.session_state.last_rag_query}'*")

        # Confidence badge
        conf_style = "badge-status-ok" if rag_res["retrieval_confidence"] in ["High", "Moderate"] else "badge-status-warn"
        st.markdown(
            f'<div class="{conf_style}">Retrieval Confidence: <strong>{rag_res["retrieval_confidence"]}</strong> '
            f'(Max Cosine Similarity: {rag_res["max_similarity_score"]:.3f}) | Mode: <code>{rag_res["mode"]}</code></div>',
            unsafe_allow_html=True
        )

        st.markdown("#### Grounded Recommendation")
        st.markdown(rag_res["answer"])

        if rag_res["citations"]:
            st.markdown("#### Document Citations")
            for cit in rag_res["citations"]:
                st.markdown(f"- 📄 `{cit}`")

        # Inspect retrieved raw chunks
        with st.expander("🔍 Inspect Retrieved Knowledge Base Passages"):
            for idx, p in enumerate(rag_res["retrieved_passages"]):
                st.markdown(f"**Chunk #{idx+1}: {p['document']} — Section: {p['section']}** (Similarity: `{p['score']:.4f}`)")
                st.text(p["text"])
                st.markdown("---")

    # Upload custom document to knowledge base
    st.markdown("---")
    with st.expander("📂 Ingest Additional Documents into Knowledge Base"):
        kb_upload = st.file_uploader("Upload custom agronomic TXT document", type=["txt"])
        if kb_upload is not None:
            content_str = kb_upload.getvalue().decode("utf-8", errors="ignore")
            extra = [(kb_upload.name, content_str)]
            num_chunks = rag_engine.reload_and_reindex(extra_texts=extra)
            st.success(f"Successfully ingested `{kb_upload.name}`! Knowledge base now contains {num_chunks} chunks.")

# =========================================================
# TAB 5: REPORTS & HISTORY
# =========================================================
with tabs[4]:
    st.subheader("Field Inspection Reports & Session History")
    
    st.markdown("### Generate Comprehensive HTML Report")
    st.markdown("Produce a print-ready, auditable HTML report compiling model configuration, detections, XAI heatmap, and RAG recommendations.")

    cur_detections = st.session_state.detection_results["detections"] if st.session_state.detection_results else []
    cur_annot_img = st.session_state.detection_results["annotated_image"] if st.session_state.detection_results else None

    html_report_code = SugarcaneReportGenerator.generate_html_report(
        image_name=st.session_state.current_image_name,
        model_name=os.path.basename(selected_model_path),
        class_names=detector.class_names,
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
        detections=cur_detections,
        annotated_image=cur_annot_img,
        xai_result=st.session_state.xai_result,
        rag_result=st.session_state.last_rag_result,
        rag_query=st.session_state.last_rag_query
    )

    btn_download_report = st.download_button(
        label="📄 Download Complete Field Analysis Report (HTML)",
        data=html_report_code.encode("utf-8"),
        file_name=f"SugarcaneAI_Report_{time.strftime('%Y%m%d_%H%M%S')}.html",
        mime="text/html"
    )

    with st.expander("👁️ Preview HTML Report"):
        st.components.v1.html(html_report_code, height=650, scrolling=True)

    st.markdown("---")
    st.markdown("### Current Session Analysis History")
    if st.session_state.session_history:
        history_df = pd.DataFrame(st.session_state.session_history)
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No analyses recorded in this session yet. Run detections in Tab 2 to populate history.")

# =========================================================
# TAB 6: ABOUT THE MODEL
# =========================================================
with tabs[5]:
    st.subheader("Model Specifications, Methodology, & Agronomic Disclaimers")

    st.markdown("""
    ### 1. Computer Vision Detection (YOLOv8n)
    - **Architecture:** YOLOv8 Nano (`yolov8n.yaml`) trained on merged Roboflow sugarcane gap datasets.
    - **Hyperparameters:** Trained 150 epochs, patience 30, image dimension 640x640, YOLOv8 format splits.
    - **Weight Loading Policy:** The system strictly loads existing user checkpoints (`best.pt`) without triggering fine-tuning, retraining, or replacing weights with stock models.
    - **Dynamic Discovery:** Classes are loaded dynamically from model metadata:
    """)
    if detector.is_loaded:
        st.json(detector.class_names)
    else:
        st.caption("Class dictionary will be displayed once weights are loaded.")

    st.markdown("""
    ### 2. Explainable AI Methodology
    - **Perturbation-Based Occlusion:** Systematically masks image regions with neutral color patches and measures the empirical loss of confidence on the matched detection.
    - **Matching Criterion:** Uses class consistency and bounding-box Intersection-over-Union (IoU) to avoid false matches.
    - **Attribution Interpretation:** Reveals what pixel structures the YOLOv8 neural network leveraged to confirm the bounding box. It does **not** imply biological causation.

    ### 3. Precision Agriculture Limitations
    - **Perspective & Pixel Dimensions:** Bounding box pixel length cannot be translated directly to real-world ground meters without ground sample distance (GSD) calibration and terrain orthomosaics.
    - **False Positives/Negatives:** Cloud shadows, furrow undulations, and weed canopy can mislead visual detectors.
    - **Verification Requirement:** Never order full field replanting or pesticide applications based solely on aerial bounding boxes. Always pair vision outputs with ground scouting.
    """)

st.markdown("---")
st.caption("SugarcaneAI • Developed for University Major Project Demonstration • All analytics computed from active model outputs.")
