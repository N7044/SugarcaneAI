# SugarcaneAI: An Explainable AI & Retrieval-Augmented Generation System for Sugarcane Gap Detection & Agricultural Decision Support

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.52+-red.svg)](https://streamlit.io/)
[![XAI](https://img.shields.io/badge/XAI-Occlusion%20Sensitivity-purple.svg)]()
[![RAG](https://img.shields.io/badge/RAG-TF--IDF%20%2B%20Gemini-orange.svg)]()

SugarcaneAI is an end-to-end, explainable artificial intelligence (XAI) and retrieval-augmented generation (RAG) platform tailored for early-stage sugarcane stand establishment, canopy gap detection, and grounded agronomic decision support.

---

## 🌾 Project Architecture & Core Features

```
SugarcaneAI/
├── app.py                     # Interactive Streamlit Dashboard & Analytics UI
├── detector.py                # Safe YOLOv8 weight loader & inference engine
├── explainability.py          # Empirical Occlusion Sensitivity & SHAP extension point
├── rag_engine.py              # TF-IDF Cosine Retrieval & Gemini grounded synthesis
├── report_generator.py        # Print-ready HTML report compiler
├── requirements.txt           # Verified Python dependency specification
├── README.md                  # System manual, testing, & presentation script
├── .env.example               # Environment variable templates
├── .gitignore                 # Security & artifact exclusions
├── models/
│   └── best.pt                # Trained YOLOv8 checkpoint (place your weights here)
├── knowledge_base/
│   ├── sugarcane_guidelines.txt  # Agronomy & establishment reference
│   └── gap_management.txt        # Ground verification & vision limitations
├── sample_data/
│   └── sample_field.jpg       # Built-in synthetic field validation image
└── outputs/                   # Export directory for CSVs and reports
```

### 1. Feature A — Computer Vision Detection (YOLOv8)
- Loads your pre-trained `best.pt` weights directly via Ultralytics without triggering retraining or overwriting weights.
- Dynamically extracts class mappings (`model.names`) from checkpoint metadata.
- Configurable confidence threshold, NMS IoU threshold, and inference image resolution (320-960px).
- Generates side-by-side comparative views, bounding boxes, detection dimensions, and exportable CSV tables.
- Safely handles images with zero detections without crashing.

### 2. Feature B — Explainable AI (Occlusion Sensitivity)
- Real perturbation-based XAI specifically engineered for object detection.
- Masks square spatial patches (neutral gray or soil brown) across a sliding grid.
- Tracks scalar confidence degradation for the matched target bounding box using IoU alignment and class consistency.
- Computes empirical sensitivity drops ($\Delta S = \max(0, S_{\text{orig}} - S_{\text{perturbed}})$) and overlays normalized heatmaps.
- Includes documented architectural extension specifications for SHAP Image Explainer.

### 3. Feature C — Retrieval-Augmented Generation (RAG)
- 100% local document chunking and vector space indexing using TF-IDF and Cosine Similarity over `knowledge_base/`.
- Computes real cosine similarity scores and retrieval confidence indicators.
- Seamlessly falls back to **Retrieval-Only Mode** when no LLM API key is provided, ensuring zero external dependencies for local inspection.
- When configured with a Google Gemini API key, synthesizes strictly grounded answers with traceable document and section citations.

### 4. Feature D & E — Analytics Dashboard & Downloadable Reports
- Real-time Plotly charts: class distribution histograms, confidence spread, and spatial dimensions.
- Download annotated PNG images and structured CSV detection tables.
- Standalone, print-ready HTML analysis reports with embedded base64 imagery, methodology disclosures, and agronomic disclaimers.

---

## 🚀 Installation & Environment Setup

### Prerequisites
- Python 3.10 to 3.13
- Operating System: Windows 10/11, Linux, or macOS

### Windows PowerShell Setup
```powershell
# 1. Navigate to D: drive project directory
cd D:\SugarcaneAI

# 2. (Optional) Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure Environment Variables (Optional for LLM synthesis)
Copy-Item .env.example .env
# Edit .env and paste your GEMINI_API_KEY if desired
```

### Google Colab / Linux Setup
```bash
git clone <repo_url> SugarcaneAI
cd SugarcaneAI
pip install -r requirements.txt
streamlit run app.py & npx localtunnel --port 8501
```

---

## 📦 Model Weights Placement

Place your trained YOLOv8 sugarcane-gap model weights file in the `models/` directory:
```
D:\SugarcaneAI\models\best.pt
```

Alternatively, you can:
1. Provide any custom path in the sidebar input field.
2. Click **Upload .pt Weights** in the sidebar to upload your file directly through the web UI.

> **Integrity Guarantee:** The system will **never** retrain the model, overwrite your weights, or substitute a stock model.

---

## 💻 Running the Application

Launch the Streamlit dashboard:
```powershell
streamlit run app.py
```
The application will launch automatically in your browser at `http://localhost:8501`.

---

## 🧪 Automated Testing & Smoke Test Procedure

### 1. Run Automated Unit Tests
```powershell
python -m unittest discover -s "D:\SugarcaneAI\tests"
```
Verifies IoU computation, detection DataFrame serialization, TF-IDF vector retrieval, and HTML report compilation.

### 2. Manual Verification Workflow
1. **Overview Tab:** Confirm the model status badge displays your weights file and detected class names.
2. **Gap Detection Tab:** Select *Use Built-in Sample Field Image* (or upload a field photo) and click **Run YOLOv8 Gap Detection**. Verify bounding boxes, metric cards, Plotly charts, and CSV download.
3. **Explainability Lab:** Select a detected bounding box and click **Compute Occlusion Sensitivity Heatmap**. Observe the progress bar and inspect the resulting heatmap overlay.
4. **Agriculture RAG Assistant:** Ask a sample inquiry (e.g., *"What are the agronomic thresholds for replanting?"*). Observe retrieved knowledge chunks, similarity scores, and grounded answers.
5. **Reports & History:** Click **Preview HTML Report** and verify that all metrics, imagery, and disclaimers are compiled cleanly into the downloadable HTML document.

---

## 🎓 University Major-Project Presentation / Viva Defense Script

### Slide 1: Problem Definition & Motivation
> "In early-stage sugarcane farming, germination failure causes irregular row gaps. If unaddressed within 30 to 45 days after planting, these gaps result in permanent stand deficit, aggressive weed competition, and lost cane tonnage. However, farmers need explainable, actionable intelligence—not a black-box detector."

### Slide 2: Computer Vision Core
> "We deployed a YOLOv8 nano model trained on merged Roboflow sugarcane gap imagery. Our implementation dynamically discovers model class definitions, prevents unauthorized weight overwrites, and post-processes bounding boxes into spatial dimension metrics."

### Slide 3: Genuine Explainability (XAI)
> "Rather than displaying generic decorative gradient maps, our Explainability Lab runs empirical Occlusion Sensitivity. By masking spatial regions with neutral patches and tracking class-consistent IoU bounding box degradation, we quantify the exact pixel regions that provided positive evidence for each detection."

### Slide 4: Evidence-Grounded Agronomic RAG
> "Object detection alone cannot tell a farmer what action to take. Our local TF-IDF vector retrieval pairs detected symptoms with established agronomic literature. When backed by Gemini, the system enforces strict prompt grounding, producing traceable citations while clearly distinguishing visual symptoms from subterranean causes."

### Slide 5: Methodological Honesty & Limitations
> "We explicitly disclaim that 2D image pixel bounding boxes equal ground meters or exact dead plant counts without ground-control calibration (GSD). SugarcaneAI serves as a geospatial triage tool to dispatch scouting teams before major capital expenditures are made."

---

## 🛡️ License & Disclaimers
This software is provided for educational, research, and agronomic triage evaluation. Field interventions should always be validated through physical inspection.
