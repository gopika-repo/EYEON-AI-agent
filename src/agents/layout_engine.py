import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

class LayoutEngine:
    def __init__(self):
        print("📐 Initializing YOLOv11 Layout Engine...")
        # Load the model
        self.model = YOLO('yolo11n.pt') 

    def analyze_layout(self, image: Image.Image):
        """
        Detects regions (Tables, Figures) in the document image.
        """
        # Convert PIL Image to OpenCV format (numpy array)
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Run YOLO Inference
        results = self.model(img_cv, verbose=False)[0]
        
        layout_data = {
            "tables": [],
            "figures": [],
            "text_blocks": []
        }
        
        # Process detections
        for box in results.boxes:
            coords = box.xyxy[0].tolist()
            confidence = float(box.conf)
            class_id = int(box.cls)
            label = self.model.names[class_id]
            
            detection = {
                "box": [round(x) for x in coords],
                "confidence": round(confidence, 2),
                "label": label
            }
            
            layout_data["text_blocks"].append(detection)

        # Plot results
        plotted_image = results.plot()
        
        return layout_data, plotted_image