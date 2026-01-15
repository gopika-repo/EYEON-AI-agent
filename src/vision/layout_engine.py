import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict

class OmniLayoutEngine:
    def __init__(self, model_path: str = "models/yolo11n_doc_layout.pt"):
        # Load the specialized YOLOv11 model
        self.model = YOLO(model_path)
        
    def analyze_page(self, image_path: str) -> Dict[str, List]:
        """Performs structural detection on a single page image."""
        results = self.model(image_path)[0]
        
        # We categorize findings into a clean structure for our Agents
        structured_map = {
            "tables": [],
            "figures": [],
            "headers": [],
            "text_blocks": []
        }
        
        for box in results.boxes:
            cls = int(box.cls[0])
            label = results.names[cls]
            coords = box.xyxy[0].tolist() # [x1, y1, x2, y2]
            conf = float(box.conf[0])
            
            element = {"bbox": coords, "confidence": conf}
            
            # Map YOLO labels to our Agentic Schema
            if "Table" in label:
                structured_map["tables"].append(element)
            elif "Picture" in label or "Figure" in label:
                structured_map["figures"].append(element)
            elif "Title" in label or "Header" in label:
                structured_map["headers"].append(element)
            else:
                structured_map["text_blocks"].append(element)
                
        return structured_map

    def crop_element(self, image_np: np.ndarray, bbox: List[float]):
        """Helper to crop detected elements for OCR or sub-analysis."""
        x1, y1, x2, y2 = map(int, bbox)
        return image_np[y1:y2, x1:x2]