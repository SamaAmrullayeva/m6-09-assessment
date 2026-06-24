"""
detector.py — CatDetector wraps the ONNX model and returns bounding boxes
in original-image pixel coordinates.

Output format per detection:
  {"xmin": float, "ymin": float, "xmax": float, "ymax": float,
   "confidence": float, "class": str}
"""

import numpy as np
import onnxruntime as ort
from PIL import Image


class CatDetector:
    def __init__(
        self,
        onnx_path: str,
        imgsz: int = 640,
        conf: float = 0.25,
        class_names: tuple = ("cat",),
    ):
        self.session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self.imgsz       = imgsz
        self.conf        = conf
        self.class_names = class_names
        self.input_name  = self.session.get_inputs()[0].name

        # Log output shape so mismatches are caught early
        out_shape = self.session.get_outputs()[0].shape
        print(f"[detector] ONNX output shape: {out_shape}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, image_path: str) -> list[dict]:
        """
        Run inference on one image file.

        Returns a list of dicts:
          [{"xmin", "ymin", "xmax", "ymax", "confidence", "class"}, ...]

        An empty list means no detections above self.conf threshold.
        """
        img_pil = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img_pil.size

        # 1. Letterbox resize to (imgsz × imgsz), record transform params
        lb_img, scale, (pad_x, pad_y) = self._letterbox(img_pil, self.imgsz)

        # 2. HWC → CHW float32 normalised to [0,1], add batch dim
        x = (np.array(lb_img, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None, ...]

        # 3. Run ONNX session
        raw = self.session.run(None, {self.input_name: x})[0]

        # YOLO26 end-to-end (NMS-free) output: (1, 300, 6)
        #   6 values per box: [x1, y1, x2, y2, score, class_id]
        # If your export shape differs (e.g. legacy (1, nc+4, 8400)),
        # you will need to add NMS — strongly recommended to keep end2end.
        detections = raw[0]  # (300, 6)

        results = []
        for x1, y1, x2, y2, score, cls in detections:
            if float(score) < self.conf:
                continue

            # 4. Undo letterbox: input-space pixels → original-image pixels
            x1 = (float(x1) - pad_x) / scale
            y1 = (float(y1) - pad_y) / scale
            x2 = (float(x2) - pad_x) / scale
            y2 = (float(y2) - pad_y) / scale

            # 5. Clip to image bounds
            x1 = max(0.0, min(float(orig_w), x1))
            y1 = max(0.0, min(float(orig_h), y1))
            x2 = max(0.0, min(float(orig_w), x2))
            y2 = max(0.0, min(float(orig_h), y2))

            cls_idx  = int(cls)
            cls_name = (
                self.class_names[cls_idx]
                if cls_idx < len(self.class_names)
                else str(cls_idx)
            )

            results.append(
                {
                    "xmin":       x1,
                    "ymin":       y1,
                    "xmax":       x2,
                    "ymax":       y2,
                    "confidence": float(score),
                    "class":      cls_name,
                }
            )

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _letterbox(img_pil: Image.Image, new_size: int):
        """
        Resize image preserving aspect ratio, pad to new_size × new_size
        with grey (114, 114, 114).

        Returns:
          canvas   — PIL Image, new_size × new_size
          scale    — float, resize scale factor applied
          (pad_x, pad_y) — pixel offsets of the pasted image inside canvas
        """
        orig_w, orig_h = img_pil.size
        scale  = min(new_size / orig_w, new_size / orig_h)
        new_w  = int(orig_w * scale)
        new_h  = int(orig_h * scale)
        resized = img_pil.resize((new_w, new_h), Image.BILINEAR)
        pad_x  = (new_size - new_w) // 2
        pad_y  = (new_size - new_h) // 2
        canvas = Image.new("RGB", (new_size, new_size), (114, 114, 114))
        canvas.paste(resized, (pad_x, pad_y))
        return canvas, scale, (pad_x, pad_y)
