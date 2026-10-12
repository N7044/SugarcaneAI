"""
explainability.py - Explainable AI (XAI) Module for Sugarcane Gap Detection
Implements genuine Occlusion Sensitivity Analysis for object detection models,
measuring empirical confidence drops under systematic spatial masking.
Includes documented SHAP extension points and rigorous attribution metrics.
"""

import time
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
import cv2
from PIL import Image


def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    """
    Computes Intersection over Union (IoU) between two bounding boxes:
    box = [x1, y1, x2, y2]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_width = max(0.0, xB - xA)
    inter_height = max(0.0, yB - yA)
    inter_area = inter_width * inter_height

    boxA_area = max(0.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxB_area = max(0.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    union_area = boxA_area + boxB_area - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


class OcclusionSensitivityExplainer:
    """
    Empirical Occlusion Sensitivity Explainer for YOLOv8 Object Detectors.

    Methodology:
    1. Given a target detection (class_id, original_bbox, original_confidence S_orig).
    2. Slides an occlusion mask (patch_size x patch_size, neutral gray or mean background)
       across the image with step size = stride.
    3. For each patch position (px, py), re-runs model inference on the perturbed image.
    4. Evaluates all perturbed detections, filtering for matching class and maximum IoU
       relative to the original bounding box.
    5. Target scalar score S_pert is the confidence of the matched detection (or 0.0 if suppressed).
    6. Sensitivity score is defined as Delta = max(0, S_orig - S_pert), representing how much
       obscuring that specific region weakened the detector evidence for this gap.
    7. Assembles a 2D spatial sensitivity grid, normalizes it to [0, 1], and produces
       an overlay heatmap.
    """

    def __init__(self, detector):
        self.detector = detector

    def explain_detection(
        self,
        image: Image.Image,
        target_detection: Dict[str, Any],
        patch_size: int = 48,
        stride: int = 24,
        mask_color: Tuple[int, int, int] = (128, 128, 128),
        iou_match_threshold: float = 0.25,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Runs occlusion sensitivity analysis for a chosen target detection.

        Args:
            image: Original PIL Image.
            target_detection: Selected detection dict from detector.predict().
            patch_size: Size of the square occlusion patch in pixels.
            stride: Stride step in pixels for the sliding occlusion window.
            mask_color: RGB tuple for the occluding patch (default neutral gray 128,128,128).
            iou_match_threshold: Minimum IoU to consider a perturbed detection a match.
            progress_callback: Optional callback receiving (fraction [0..1], status_text).

        Returns:
            Dict containing:
            - 'heatmap_overlay': PIL Image with heatmap blended on original.
            - 'raw_heatmap': 2D float numpy array [H, W] normalized [0, 1].
            - 'original_score': Original confidence score.
            - 'min_perturbed_score': Lowest confidence observed under occlusion.
            - 'mean_perturbed_score': Mean confidence across all perturbations.
            - 'max_sensitivity_drop': Maximum drop (S_orig - S_min).
            - 'total_evaluations': Number of masked inferences executed.
            - 'runtime_sec': Total execution time.
            - 'method_explanation': Plain-language methodology summary.
        """
        start_time = time.time()
        orig_img_np = np.array(image.convert('RGB'))
        H, W, _ = orig_img_np.shape

        target_box = [
            float(target_detection['xmin']),
            float(target_detection['ymin']),
            float(target_detection['xmax']),
            float(target_detection['ymax'])
        ]
        target_cls = int(target_detection['class_id'])
        target_score = float(target_detection['confidence'])

        # Create sampling grid coordinates
        y_coords = list(range(0, H, stride))
        x_coords = list(range(0, W, stride))
        total_steps = len(y_coords) * len(x_coords)

        # Sensitivity accumulator and hit counter matrices
        sensitivity_grid = np.zeros((H, W), dtype=np.float32)
        count_grid = np.zeros((H, W), dtype=np.float32)

        perturbed_scores: List[float] = []
        eval_counter = 0

        for y in y_coords:
            for x in x_coords:
                eval_counter += 1
                if progress_callback and (eval_counter % max(1, total_steps // 20) == 0 or eval_counter == total_steps):
                    prog = eval_counter / total_steps
                    progress_callback(prog, f"Evaluating occlusion patch {eval_counter}/{total_steps}...")

                # Create perturbed image
                x1 = x
                y1 = y
                x2 = min(W, x + patch_size)
                y2 = min(H, y + patch_size)

                perturbed_img = orig_img_np.copy()
                perturbed_img[y1:y2, x1:x2] = mask_color

                # Inference on perturbed image
                res = self.detector.model.predict(
                    source=perturbed_img,
                    conf=0.05,  # Low threshold to capture degraded confidence
                    iou=0.45,
                    imgsz=640,
                    verbose=False
                )

                # Find best matching detection for target_detection
                matched_score = 0.0
                best_match_iou = 0.0

                if len(res) > 0 and len(res[0].boxes) > 0:
                    boxes = res[0].boxes
                    for b in boxes:
                        cls_id = int(b.cls[0].item())
                        if cls_id == target_cls:
                            b_xyxy = b.xyxy[0].cpu().numpy().tolist()
                            iou_val = compute_iou(target_box, b_xyxy)
                            if iou_val > iou_match_threshold and iou_val > best_match_iou:
                                best_match_iou = iou_val
                                matched_score = float(b.conf[0].item())

                perturbed_scores.append(matched_score)

                # Positive sensitivity: how much did occlusion decrease confidence?
                drop = max(0.0, target_score - matched_score)

                # Accumulate drop over the patched region
                sensitivity_grid[y1:y2, x1:x2] += drop
                count_grid[y1:y2, x1:x2] += 1.0

        if progress_callback:
            progress_callback(1.0, "Finalizing attribution heatmap...")

        # Normalize sensitivity grid by counts
        valid_counts = np.maximum(count_grid, 1.0)
        mean_sensitivity = sensitivity_grid / valid_counts

        # Smooth and normalize to [0, 1]
        sigma = max(1, patch_size // 4)
        smoothed = cv2.GaussianBlur(mean_sensitivity, (0, 0), sigmaX=sigma, sigmaY=sigma)
        max_val = float(np.max(smoothed))
        if max_val > 1e-6:
            norm_heatmap = smoothed / max_val
        else:
            norm_heatmap = smoothed

        # Create overlay visualization using Jet colormap
        heatmap_uint8 = np.uint8(255 * norm_heatmap)
        colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        colormap_rgb = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)

        # Alpha blend with original image (alpha 0.45)
        overlay = cv2.addWeighted(orig_img_np, 0.55, colormap_rgb, 0.45, 0)

        # Draw original target bounding box on top in bright yellow for spatial reference
        x1_t, y1_t, x2_t, y2_t = [int(v) for v in target_box]
        cv2.rectangle(overlay, (x1_t, y1_t), (x2_t, y2_t), (255, 255, 0), 2, lineType=cv2.LINE_AA)
        cv2.putText(
            overlay,
            f"Target #{target_detection.get('detection_id', 1)} ({target_score:.2f})",
            (x1_t + 4, max(18, y1_t - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            lineType=cv2.LINE_AA
        )

        runtime = round(time.time() - start_time, 2)
        min_p = float(np.min(perturbed_scores)) if perturbed_scores else target_score
        mean_p = float(np.mean(perturbed_scores)) if perturbed_scores else target_score

        method_text = (
            "Occlusion Sensitivity tests empirical model reliance on specific image regions. "
            f"A {patch_size}x{patch_size}px neutral patch systematically masked {total_steps} positions across the frame. "
            "Warmer regions (red/yellow) indicate areas where masking caused the sharpest decrease in detection "
            f"confidence (dropping from {target_score:.3f} down to {min_p:.3f}). "
            "Important note: This reveals functional input sensitivity of the YOLOv8 detector; it does not prove "
            "biological causation or crop mortality factors."
        )

        return {
            "heatmap_overlay": Image.fromarray(overlay),
            "raw_heatmap": norm_heatmap,
            "original_score": target_score,
            "min_perturbed_score": round(min_p, 4),
            "mean_perturbed_score": round(mean_p, 4),
            "max_sensitivity_drop": round(target_score - min_p, 4),
            "total_evaluations": total_steps,
            "runtime_sec": runtime,
            "method_explanation": method_text,
            "target_box": target_box
        }


class SHAPExtensionPoint:
    """
    Documented Architectural Extension Point for SHAP (SHapley Additive exPlanations).

    In object detection, applying standard SHAP directly to raw multi-head output tensors
    (bounding boxes + class logits across hundreds of anchor cells) leads to ill-defined
    value functions unless reduced to a scalar detection objective.

    This extension point defines the required interface for integrating SHAP PartitionExplainer
    or KernelExplainer against a single scalar detection score function.
    """

    @staticmethod
    def get_extension_documentation() -> Dict[str, Any]:
        return {
            "status": "Documented Extension Point",
            "recommended_algorithm": "shap.explainers.PartitionExplainer or KernelExplainer",
            "scalar_objective": "f(mask) -> Matched confidence score of target gap detection",
            "computational_profile": "High: typically requires 500-2000 evaluations per detection",
            "interface_spec": (
                "To activate full SHAP: install shap>=0.44.0, wrap the YOLOv8 image inference "
                "in a function f(image_mask) -> scalar_score, and initialize "
                "shap.Explainer(f, shap.maskers.Image('blur(32, 32)', shape=(H, W, 3)))."
            )
        }
