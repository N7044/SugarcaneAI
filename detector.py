"""
detector.py - YOLOv8 Object Detection Module for SugarcaneAI
Loads pre-trained YOLOv8 weights (best.pt), performs robust inference,
extracts dynamic class mappings, and provides clean post-processing.
"""

import os
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from ultralytics import YOLO


class SugarcaneDetector:
    """
    YOLOv8 Sugarcane Gap Detector wrapper.
    Ensures safe loading of user-provided trained weights without retraining
    or overwriting existing model checkpoints.
    """

    def __init__(self, model_path: str):
        self.model_path = os.path.abspath(model_path)
        self.model: Optional[YOLO] = None
        self.class_names: Dict[int, str] = {}
        self.load_error: Optional[str] = None
        self._load_model()

    def _load_model(self) -> None:
        """Loads the existing YOLO weights with rigorous error handling."""
        if not os.path.exists(self.model_path):
            self.load_error = (
                f"Model file not found at '{self.model_path}'. "
                "Please place your trained 'best.pt' weights in 'models/' or select a valid file."
            )
            return

        try:
            # Load pre-trained weights directly
            self.model = YOLO(self.model_path)
            # Dynamically read class names from trained checkpoint
            if hasattr(self.model, 'names') and self.model.names:
                self.class_names = dict(self.model.names)
            else:
                self.class_names = {0: "gap"}
            self.load_error = None
        except Exception as e:
            self.model = None
            self.load_error = f"Failed to load YOLO model from '{self.model_path}': {str(e)}"

    @property
    def is_loaded(self) -> bool:
        """Returns True if weights were loaded successfully."""
        return self.model is not None and self.load_error is None

    def predict(
        self,
        image: Image.Image,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640
    ) -> Dict[str, Any]:
        """
        Runs inference on an input image.

        Args:
            image: PIL Image
            conf_threshold: Minimum confidence score [0.0 - 1.0]
            iou_threshold: NMS IoU threshold [0.0 - 1.0]
            imgsz: Inference image size (default 640)

        Returns:
            Dictionary containing:
            - 'detections': List of detection dictionaries
            - 'annotated_image': PIL Image with drawn bounding boxes
            - 'class_counts': Dict of class_name -> count
            - 'stats': Confidence statistics (mean, min, max)
        """
        if not self.is_loaded:
            raise RuntimeError(self.load_error or "Detector model is not loaded.")

        # Convert PIL to RGB numpy array
        img_np = np.array(image.convert("RGB"))

        # Run inference using Ultralytics YOLO
        results = self.model.predict(
            source=img_np,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            verbose=False
        )

        detections: List[Dict[str, Any]] = []
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for idx, box in enumerate(boxes):
                xyxy = box.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
                cls_id = int(box.cls[0].item())
                cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
                conf = float(box.conf[0].item())

                width_px = max(0.0, xyxy[2] - xyxy[0])
                height_px = max(0.0, xyxy[3] - xyxy[1])
                area_px = width_px * height_px

                detections.append({
                    "detection_id": idx + 1,
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf, 4),
                    "xmin": round(xyxy[0], 2),
                    "ymin": round(xyxy[1], 2),
                    "xmax": round(xyxy[2], 2),
                    "ymax": round(xyxy[3], 2),
                    "width_px": round(width_px, 2),
                    "height_px": round(height_px, 2),
                    "area_px": round(area_px, 2)
                })

        # Render clean annotated image
        annotated_img = self._annotate_image(img_np, detections)

        # Compute summary metrics
        class_counts: Dict[str, int] = {}
        for d in detections:
            c = d["class_name"]
            class_counts[c] = class_counts.get(c, 0) + 1

        confs = [d["confidence"] for d in detections]
        stats = {
            "total": len(detections),
            "mean_conf": round(float(np.mean(confs)), 4) if confs else 0.0,
            "min_conf": round(float(np.min(confs)), 4) if confs else 0.0,
            "max_conf": round(float(np.max(confs)), 4) if confs else 0.0,
        }

        return {
            "detections": detections,
            "annotated_image": annotated_img,
            "class_counts": class_counts,
            "stats": stats
        }

    def _annotate_image(self, img_np: np.ndarray, detections: List[Dict[str, Any]]) -> Image.Image:
        """Draws bounding boxes and labels cleanly with OpenCV."""
        annotated = img_np.copy()
        
        # Color palette for classes (BGR)
        palette = [
            (34, 139, 34),    # Forest green
            (0, 140, 255),    # Orange/amber
            (255, 69, 0),     # Red-orange
            (148, 0, 211),    # Violet
            (0, 191, 255),    # Deep Sky Blue
        ]

        for d in detections:
            x1, y1 = int(d["xmin"]), int(d["ymin"])
            x2, y2 = int(d["xmax"]), int(d["ymax"])
            cls_id = d["class_id"]
            color = palette[cls_id % len(palette)]

            # Draw bounding box rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)

            # Label text
            label = f"{d['class_name']} #{d['detection_id']}: {d['confidence']:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Draw background tag
            tag_y1 = max(0, y1 - th - 6)
            tag_y2 = y1
            cv2.rectangle(annotated, (x1, tag_y1), (x1 + tw + 6, tag_y2), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 3, tag_y2 - 4),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                lineType=cv2.LINE_AA
            )

        return Image.fromarray(annotated)

    @staticmethod
    def to_dataframe(detections: List[Dict[str, Any]]) -> pd.DataFrame:
        """Converts detection list into a clean Pandas DataFrame for display and CSV export."""
        if not detections:
            return pd.DataFrame(columns=[
                "ID", "Class", "Confidence", "Xmin", "Ymin", "Xmax", "Ymax", "Width(px)", "Height(px)", "Area(px)"
            ])
        df = pd.DataFrame(detections)
        rename_cols = {
            "detection_id": "ID",
            "class_name": "Class",
            "confidence": "Confidence",
            "xmin": "Xmin",
            "ymin": "Ymin",
            "xmax": "Xmax",
            "ymax": "Ymax",
            "width_px": "Width(px)",
            "height_px": "Height(px)",
            "area_px": "Area(px)"
        }
        df = df[[c for c in rename_cols.keys() if c in df.columns]].rename(columns=rename_cols)
        return df
