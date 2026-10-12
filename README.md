# 🌱 SugarcaneAI: Explainable AI & Retrieval-Augmented Generation for Sugarcane Gap Detection

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-2b5b84.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics%208.4+-00ffff.svg)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.52+-ff4b4b.svg)](https://streamlit.io/)
[![XAI](https://img.shields.io/badge/XAI-Occlusion%20Sensitivity-7c3aed.svg)]()
[![RAG](https://img.shields.io/badge/RAG-TF--IDF%20%2B%20Gemini-10b981.svg)]()
[![License](https://img.shields.io/badge/License-MIT-gray.svg)]()

**SugarcaneAI** is an end-to-end, explainable artificial intelligence (XAI) and retrieval-augmented generation (RAG) system engineered for precision agriculture, sugarcane crop stand evaluation, canopy gap detection, and agronomic decision support.

---

## 📌 Table of Contents
1. [Overview & Key Capabilities](#-overview--key-capabilities)
2. [Trained Model Weights (`best.pt`)](#-trained-model-weights-bestpt)
3. [Project Structure](#-project-structure)
4. [Quickstart & How to Run](#-quickstart--how-to-run)
5. [Feature Guide](#-feature-guide)
   - [Feature A: Computer Vision Detection](#feature-a--computer-vision-detection)
   - [Feature B: Explainable AI (Occlusion Sensitivity)](#feature-b--explainable-ai-occlusion-sensitivity)
   - [Feature C: Agronomic Knowledge Assistant (RAG)](#feature-c--agronomic-knowledge-assistant-rag)
   - [Feature D: Real-Time Field Analytics](#feature-d--real-time-field-analytics)
   - [Feature E: Auditable HTML Field Reports](#feature-e--auditable-html-field-reports)
6. [Testing & Verification](#-testing--verification)
7. [Academic Viva & Presentation Script](#-academic-viva--presentation-script)
8. [Agronomic Disclaimers](#-agronomic-disclaimers)

---

## 🌟 Overview & Key Capabilities

In early-stage sugarcane cultivation, uneven germination results in canopy voids (gaps). Unaddressed gaps (> 0.5–1.0 meter) compromise plant population density, invite weed infestation, waste solar radiation, and permanently reduce harvest tonnage.

SugarcaneAI solves this by uniting:
- **Object Detection (YOLOv8n):** Autonomous visual indexing of crop canopy gaps from aerial drone and field imagery.
- **Empirical Explainability (Occlusion Sensitivity):** Perturbation testing that quantifies how much image regions drive model confidence, eliminating "black-box" uncertainty.
- **Evidence-Grounded RAG:** Local TF-IDF vector retrieval and Google Gemini LLM synthesis providing traceable agronomic recommendations with document-level citations.
- **Full Field Analytics & Reporting:** Downloadable CSV logs, annotated imagery, and self-contained HTML inspection reports.

---

## 🧠 Trained Model Weights (`best.pt`)

The system uses a custom-trained **YOLOv8n** model trained on merged Roboflow sugarcane-gap datasets.

- **Checkpoint Files:** Included directly in the repository at:
  - Root: `best.pt`
  - Models folder: `models/best.pt`
- **Model Architecture:** YOLOv8 Nano (`yolov8n.yaml`)
- **Trained Classes:** Dynamically inspected from checkpoint:
  ```json
  { 0: "gap" }
  ```
- **Integrity Guarantee:** The application **never** retrains the model during startup and never replaces existing weights with uninitialized checkpoints.

---

## 📂 Project Structure

```
SugarcaneAI/
├── best.pt                    # Trained YOLOv8 weights (root location)
├── app.py                     # Streamlit application UI & dashboard
├── detector.py                # Safe model loader, dynamic class discovery & inference
├── explainability.py          # Occlusion sensitivity XAI engine & SHAP extension point
├── rag_engine.py              # TF-IDF Cosine Retrieval & Gemini grounded synthesis
├── report_generator.py        # Print-ready HTML field report compiler
├── requirements.txt           # Verified Python dependency specifications
├── README.md                  # Comprehensive documentation and run guide
├── .env.example               # Environment variables template (API keys)
├── .gitignore                 # Exclusion rules
├── models/
│   └── best.pt                # Mirror copy of trained weights
├── knowledge_base/
│   ├── sugarcane_guidelines.txt  # Agronomy, planting dynamics & gap definitions
│   └── gap_management.txt        # Ground verification, replanting & remote sensing limits
├── sample_data/
│   └── sample_field.jpg       # Sample field test image
├── tests/
│   └── test_pipeline.py       # Automated unit test suite
└── outputs/                   # Export directory for CSVs and reports
```

---

## ⚡ Quickstart & How to Run

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13 installed.
- Git installed.

### 2. Windows PowerShell Setup
```powershell
# Clone or navigate to the project directory
cd D:\SugarcaneAI

# (Optional) Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt

# (Optional) Configure Gemini API key for synthesized LLM answers
Copy-Item .env.example .env
# Edit .env and enter GEMINI_API_KEY=your_key_here
```

### 3. Launch the Application
```powershell
streamlit run app.py
```
Your default web browser will automatically open:
```
http://localhost:8501
```

*(Note: The application operates 100% locally even without an internet connection or API key; the RAG assistant runs in local Retrieval-Only Mode).*

---

## 🔍 Feature Guide

### Feature A — Computer Vision Detection
1. Open the **Gap Detection** tab.
2. Select **Use Built-in Sample Field Image** or upload your own drone images (`.jpg`, `.jpeg`, `.png`).
3. Adjust the **Confidence Threshold** (default: `0.25`) and **NMS IoU Threshold** (default: `0.45`) in the sidebar.
4. Click **🚀 Run YOLOv8 Gap Detection**.
5. Inspect the side-by-side original and annotated views, detection statistics, and coordinate tables.
6. Export the results using **Download Annotated Image** or **Download Detections (CSV)**.

### Feature B — Explainable AI (Occlusion Sensitivity)
1. Navigate to the **Explainability Lab (XAI)** tab.
2. Select any detected bounding box from the dropdown.
3. Choose patch size (e.g., $48\text{px}$) and stride step (e.g., $24\text{px}$).
4. Click **🔬 Compute Occlusion Sensitivity Heatmap**.
5. The engine systematically occludes spatial regions, records confidence drops on the matched bounding box, and computes a smoothed Jet colormap heatmap overlay.
6. Review the target score drop, minimum score under occlusion, and attribution explanation.

### Feature C — Agronomic Knowledge Assistant (RAG)
1. Navigate to the **Agriculture RAG Assistant** tab.
2. Choose from curated prompts or type a custom inquiry:
   - *"What causes poor germination and gap formation in sugarcane?"*
   - *"When is the best time after planting to perform gap filling?"*
   - *"What are the critical limitations of drone and vision-based gap detection?"*
3. The engine performs local TF-IDF vectorization and cosine similarity matching over the documents in `knowledge_base/`.
4. Inspect the retrieved passages, similarity scores, and grounded answers with traceable citations.
5. Ingest additional custom agronomic documents live via the file uploader.

### Feature D — Real-Time Field Analytics
- Interactive Plotly visualizations including **Detection Count by Class** bar charts and **Confidence Score Distribution** histograms.
- Summary metrics: total detections, mean confidence, minimum confidence, and maximum confidence.

### Feature E — Auditable HTML Field Reports
1. Go to the **Reports & History** tab.
2. Click **📄 Download Complete Field Analysis Report (HTML)**.
3. The generated report compiles model details, detection statistics, base64-embedded images, XAI heatmaps, RAG recommendations, and agronomic disclaimers into a single, printable HTML file.

---

## 🧪 Testing & Verification

Run the automated test suite directly from your terminal:
```powershell
python -m unittest discover -s "D:\SugarcaneAI\tests"
```

**Expected output:**
```
....
----------------------------------------------------------------------
Ran 4 tests in 0.341s

OK
```

**Verified components:**
- IoU bounding box computation and overlap thresholds.
- Pandas DataFrame post-processing and coordinate formatting.
- TF-IDF indexing and cosine similarity retrieval from `knowledge_base/`.
- Full standalone HTML report compilation and rendering.

---

## 🎓 Academic Viva & Presentation Script

Use this script during project evaluations or defense presentations:

- **Problem:** *"In sugarcane agriculture, germination failure causes canopy gaps. Missing 30% of an initial stand directly depresses final cane yield and allows weed competition. Agronomists need an objective, explainable decision support tool."*
- **Detection Model:** *"We trained and loaded a YOLOv8 nano detector on sugarcane gap aerial datasets. The system dynamically reads model metadata and extracts spatial bounding coordinates."*
- **Explainable AI:** *"Rather than showing generic classification saliency maps, we implemented empirical Occlusion Sensitivity specifically adapted for object detection. It masks spatial patches and tracks confidence degradation on class-consistent, IoU-matched bounding boxes. This visually demonstrates the pixel regions the model relied on."*
- **Grounded Decision Support:** *"Detecting a gap does not explain what action to take. Our TF-IDF vector retrieval pairs detected symptoms with agronomic management guidelines, providing evidence-backed recommendations and document citations."*

---

## ⚠️ Agronomic Disclaimers

1. **Pixel vs. Metric Dimensions:** Detected 2D image pixel intervals do not represent calibrated ground meters without Ground Control Points (GCPs) and Ground Sample Distance (GSD) orthomosaic calibration.
2. **Visual Symptoms vs. Subsurface Causes:** Computer vision identifies canopy absence. It cannot diagnose subsurface factors (such as sett rot, soil salinity, compaction, or root grubs) without physical soil sampling.
3. **Scouting Protocol:** SugarcaneAI is designed as a geospatial triage system to prioritize field scouting crews before capital replanting decisions are made.
