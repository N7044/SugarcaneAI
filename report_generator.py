"""
report_generator.py - Comprehensive HTML Analysis Report Generator for SugarcaneAI
Produces standalone, beautifully styled, print-ready agronomic reports
incorporating YOLOv8 detection results, Occlusion XAI heatmaps, RAG evidence,
and rigorous agricultural methodology disclosures.
"""

import os
import io
import base64
from datetime import datetime
from typing import Dict, List, Optional, Any
from PIL import Image


def image_to_base64(img: Image.Image) -> str:
    """Encodes a PIL image to a base64 PNG data URI string."""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


class SugarcaneReportGenerator:
    """Generates standalone self-contained HTML analysis reports."""

    @staticmethod
    def generate_html_report(
        image_name: str,
        model_name: str,
        class_names: Dict[int, str],
        conf_thresh: float,
        iou_thresh: float,
        detections: List[Dict[str, Any]],
        annotated_image: Optional[Image.Image] = None,
        xai_result: Optional[Dict[str, Any]] = None,
        rag_result: Optional[Dict[str, Any]] = None,
        rag_query: Optional[str] = None
    ) -> str:
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        total_detections = len(detections)

        # Class breakdown
        class_counts: Dict[str, int] = {}
        for d in detections:
            c = d.get("class_name", "unknown")
            class_counts[c] = class_counts.get(c, 0) + 1

        confs = [d["confidence"] for d in detections]
        mean_conf = f"{sum(confs)/len(confs):.3f}" if confs else "N/A"
        min_conf = f"{min(confs):.3f}" if confs else "N/A"
        max_conf = f"{max(confs):.3f}" if confs else "N/A"

        # Embedded images
        annotated_b64 = image_to_base64(annotated_image) if annotated_image else None
        xai_b64 = image_to_base64(xai_result["heatmap_overlay"]) if (xai_result and "heatmap_overlay" in xai_result) else None

        # Build table rows
        table_rows = []
        for d in detections:
            table_rows.append(f"""
            <tr>
                <td>#{d.get('detection_id', '-')}</td>
                <td><span class="badge class-badge">{d.get('class_name', '-')}</span></td>
                <td><strong>{d.get('confidence', 0.0):.3f}</strong></td>
                <td>[{d.get('xmin', 0)}, {d.get('ymin', 0)}, {d.get('xmax', 0)}, {d.get('ymax', 0)}]</td>
                <td>{d.get('width_px', 0)} x {d.get('height_px', 0)}</td>
                <td>{d.get('area_px', 0)}</td>
            </tr>
            """)
        table_html = "".join(table_rows) if table_rows else "<tr><td colspan='6' style='text-align:center;'>No detections recorded above confidence threshold.</td></tr>"

        # Class breakdown list
        class_summary_items = [f"<li><strong>{c}</strong>: {count} detection(s)</li>" for c, count in class_counts.items()]
        class_summary_html = "".join(class_summary_items) if class_summary_items else "<li>No classes detected.</li>"

        # XAI section
        xai_section = ""
        if xai_result:
            xai_section = f"""
            <div class="card">
                <h2>Feature B: Explainable AI (Occlusion Sensitivity Analysis)</h2>
                <p>Empirical perturbation analysis measuring model confidence drops when spatial image patches are masked.</p>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{xai_result.get('original_score', 0):.3f}</div>
                        <div class="metric-label">Target Detection Score</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{xai_result.get('min_perturbed_score', 0):.3f}</div>
                        <div class="metric-label">Minimum Score Under Occlusion</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{xai_result.get('max_sensitivity_drop', 0):.3f}</div>
                        <div class="metric-label">Max Sensitivity Drop</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{xai_result.get('total_evaluations', 0)}</div>
                        <div class="metric-label">Mask Perturbations Evaluated</div>
                    </div>
                </div>
                <div style="margin-top: 15px; text-align: center;">
                    {f'<img src="{xai_b64}" alt="Occlusion Heatmap" style="max-width: 100%; border-radius: 6px; border: 1px solid #c2d6c4;" />' if xai_b64 else ''}
                </div>
                <div class="callout callout-info" style="margin-top: 12px;">
                    <strong>Explanation Summary:</strong> {xai_result.get('method_explanation', '')}
                </div>
            </div>
            """

        # RAG section
        rag_section = ""
        if rag_result and rag_query:
            citations_list = "".join([f"<li>{c}</li>" for c in rag_result.get("citations", [])])
            passages_html = ""
            for p in rag_result.get("retrieved_passages", []):
                passages_html += f"""
                <div class="passage-box">
                    <strong>Source: {p.get('document', '')} | Section: {p.get('section', '')} (Similarity: {p.get('score', 0):.3f})</strong>
                    <p>{p.get('text', '')}</p>
                </div>
                """

            rag_section = f"""
            <div class="card">
                <h2>Feature C: Agronomic Decision Support (RAG Q&A)</h2>
                <div class="callout callout-query">
                    <strong>Agronomist Inquiry:</strong> {rag_query}
                </div>
                <div style="margin: 14px 0;">
                    <strong>Synthesis Mode:</strong> <span class="badge">{rag_result.get('mode', 'N/A')}</span> | 
                    <strong>Retrieval Confidence:</strong> <span class="badge">{rag_result.get('retrieval_confidence', 'N/A')}</span> (Max Cosine Similarity: {rag_result.get('max_similarity_score', 0):.3f})
                </div>
                <div class="callout callout-success">
                    <strong>Grounded Answer / Recommendation:</strong>
                    <div style="margin-top: 6px; white-space: pre-wrap;">{rag_result.get('answer', '')}</div>
                </div>
                <h3 style="margin-top: 15px;">Document Citations:</h3>
                <ul>{citations_list or '<li>No direct citations recorded.</li>'}</ul>
                <h3 style="margin-top: 15px;">Retrieved Knowledge Base Passages:</h3>
                {passages_html}
            </div>
            """

        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>SugarcaneAI Field Analysis Report - {image_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #f4f7f4;
            color: #1c2826;
            margin: 0;
            padding: 30px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
        }}
        .header {{
            background: linear-gradient(135deg, #1b4332, #2d6a4f);
            color: #ffffff;
            padding: 28px;
            border-radius: 10px;
            margin-bottom: 24px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        .header h1 {{ margin: 0 0 8px 0; font-size: 26px; }}
        .header p {{ margin: 0; opacity: 0.9; font-size: 14px; }}
        .card {{
            background: #ffffff;
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 24px;
            border: 1px solid #d8e2dc;
            box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        }}
        h2 {{
            color: #1b4332;
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 8px;
            margin-top: 0;
            font-size: 20px;
        }}
        h3 {{ color: #2d6a4f; margin-top: 16px; font-size: 16px; }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin: 16px 0;
        }}
        .metric-card {{
            background: #f8faf8;
            border: 1px solid #d0ded1;
            padding: 16px;
            border-radius: 6px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #1b4332;
        }}
        .metric-label {{
            font-size: 13px;
            color: #556b2f;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 4px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0;
            font-size: 14px;
        }}
        th, td {{
            border: 1px solid #e2e8f0;
            padding: 10px 12px;
            text-align: left;
        }}
        th {{
            background-color: #e8f0e9;
            color: #1b4332;
            font-weight: 600;
        }}
        tr:nth-child(even) {{ background-color: #fafcfa; }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
            background: #e2e8f0;
            color: #334155;
        }}
        .class-badge {{
            background: #d8f3dc;
            color: #1b4332;
        }}
        .callout {{
            padding: 14px 18px;
            border-radius: 6px;
            margin: 14px 0;
            font-size: 14px;
        }}
        .callout-info {{ background: #eaf4ff; border-left: 4px solid #0066cc; color: #003366; }}
        .callout-warning {{ background: #fffbe6; border-left: 4px solid #faad14; color: #874d00; }}
        .callout-success {{ background: #f0fdf4; border-left: 4px solid #22c55e; color: #14532d; }}
        .callout-query {{ background: #f5f3ff; border-left: 4px solid #7c3aed; color: #4c1d95; }}
        .passage-box {{
            background: #f8faf8;
            border: 1px solid #e0eae1;
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 10px;
            font-size: 13px;
        }}
        .image-container {{
            text-align: center;
            margin: 16px 0;
        }}
        .image-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 6px;
            border: 1px solid #d0ded1;
        }}
        .footer {{
            text-align: center;
            font-size: 12px;
            color: #64748b;
            margin-top: 30px;
            padding-top: 15px;
            border-top: 1px solid #e2e8f0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>SugarcaneAI: Field Inspection & Agronomic Decision Report</h1>
            <p>Generated: {timestamp_str} | Target Image: {image_name}</p>
        </div>

        <div class="card">
            <h2>System & Model Configuration</h2>
            <table>
                <tr><th style="width:30%;">Parameter</th><th>Configuration Value</th></tr>
                <tr><td>Model Weights File</td><td><code>{model_name}</code></td></tr>
                <tr><td>Detected Classes</td><td>{", ".join([f"{cid}: {cname}" for cid, cname in class_names.items()]) if class_names else "gap"}</td></tr>
                <tr><td>Confidence Threshold</td><td>{conf_thresh:.2f}</td></tr>
                <tr><td>NMS IoU Threshold</td><td>{iou_thresh:.2f}</td></tr>
                <tr><td>Image Processing Size</td><td>640 x 640</td></tr>
            </table>
        </div>

        <div class="card">
            <h2>Feature A: Object Detection Analytics</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{total_detections}</div>
                    <div class="metric-label">Total Detections</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{mean_conf}</div>
                    <div class="metric-label">Mean Confidence</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{min_conf} - {max_conf}</div>
                    <div class="metric-label">Confidence Range</div>
                </div>
            </div>

            <h3>Class Distribution</h3>
            <ul>{class_summary_html}</ul>

            {f'<div class="image-container"><h3>Annotated Field Image</h3><img src="{annotated_b64}" alt="Annotated Sugarcane Image" /></div>' if annotated_b64 else ''}

            <h3>Detections Table</h3>
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Class</th>
                        <th>Confidence</th>
                        <th>Bounding Box [X1, Y1, X2, Y2]</th>
                        <th>Dimensions (W x H px)</th>
                        <th>Area (px)</th>
                    </tr>
                </thead>
                <tbody>
                    {table_html}
                </tbody>
            </table>
        </div>

        {xai_section}

        {rag_section}

        <div class="card">
            <h2>Agronomic Methodology, Ground Protocol, & Limitations</h2>
            <div class="callout callout-warning">
                <strong>Crucial Agricultural Notice:</strong>
                <ul>
                    <li><strong>Pixel Dimension vs Ground Length:</strong> Bounding box pixel spans do not equal calibrated physical meters unless referenced with ground control points and orthomosaic GSD metadata.</li>
                    <li><strong>No Direct Plant Mortality Diagnosis:</strong> A detected canopy gap indicates visual absence of foliage. It does not measure the exact number of failed setts or identify subsurface factors (sett rot, nematodes, compaction, soil salinity).</li>
                    <li><strong>Triage Protocol:</strong> This automated report should serve as a geospatial scouting guide to direct field personnel for manual furrow inspection prior to initiating replanting or chemical treatments.</li>
                </ul>
            </div>
        </div>

        <div class="footer">
            SugarcaneAI — Explainable AI and RAG System for Agricultural Precision Support. Generated autonomously without retraining or fabricating metrics.
        </div>
    </div>
</body>
</html>
"""
        return html_template
