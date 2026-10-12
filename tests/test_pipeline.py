"""
Unit and automated tests for SugarcaneAI core pipelines.
Verifies IoU matching, DataFrame post-processing, RAG vector retrieval, and HTML report formatting.
"""

import os
import unittest
from PIL import Image
import pandas as pd
from explainability import compute_iou
from detector import SugarcaneDetector
from rag_engine import SugarcaneRAGEngine
from report_generator import SugarcaneReportGenerator


class TestSugarcaneAIPipeline(unittest.TestCase):

    def test_iou_calculation(self):
        # Perfect overlap
        box1 = [10.0, 10.0, 50.0, 50.0]
        box2 = [10.0, 10.0, 50.0, 50.0]
        self.assertAlmostEqual(compute_iou(box1, box2), 1.0, places=4)

        # No overlap
        box3 = [100.0, 100.0, 150.0, 150.0]
        self.assertAlmostEqual(compute_iou(box1, box3), 0.0, places=4)

        # Partial overlap: [0, 0, 10, 10] and [5, 0, 15, 10]
        # Inter = 5 * 10 = 50. BoxA = 100, BoxB = 100. Union = 150. IoU = 50/150 = 1/3
        boxA = [0.0, 0.0, 10.0, 10.0]
        boxB = [5.0, 0.0, 15.0, 10.0]
        self.assertAlmostEqual(compute_iou(boxA, boxB), 1.0 / 3.0, places=4)

    def test_dataframe_conversion(self):
        detections = [
            {
                "detection_id": 1,
                "class_id": 0,
                "class_name": "gap",
                "confidence": 0.88,
                "xmin": 12.0,
                "ymin": 34.0,
                "xmax": 56.0,
                "ymax": 78.0,
                "width_px": 44.0,
                "height_px": 44.0,
                "area_px": 1936.0
            }
        ]
        df = SugarcaneDetector.to_dataframe(detections)
        self.assertEqual(len(df), 1)
        self.assertIn("ID", df.columns)
        self.assertIn("Confidence", df.columns)
        self.assertIn("Class", df.columns)

    def test_rag_retrieval(self):
        kb_path = os.path.join(os.path.dirname(__file__), "..", "knowledge_base")
        engine = SugarcaneRAGEngine(kb_path)
        self.assertTrue(engine.is_indexed)
        self.assertGreater(len(engine.chunks), 0)

        # Test query about gaps and replanting
        results = engine.retrieve("When should gap filling and replanting be done?", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertGreater(results[0]["score"], 0.0)

    def test_html_report_generation(self):
        html = SugarcaneReportGenerator.generate_html_report(
            image_name="test_field.jpg",
            model_name="best.pt",
            class_names={0: "gap"},
            conf_thresh=0.25,
            iou_thresh=0.45,
            detections=[
                {"detection_id": 1, "class_name": "gap", "confidence": 0.85, "xmin": 10, "ymin": 10, "xmax": 50, "ymax": 50, "width_px": 40, "height_px": 40, "area_px": 1600}
            ]
        )
        self.assertIn("SugarcaneAI", html)
        self.assertIn("test_field.jpg", html)
        self.assertIn("best.pt", html)


if __name__ == "__main__":
    unittest.main()
