from typing import TypedDict, List, Dict, Any
from typing_extensions import Annotated
import operator

class DocumentState(TypedDict):
    # Path to the current document being processed
    file_path: str
    poppler_path: str
    # List of images (one per page)
    pages: List[Any] 
    # The map of detected elements (tables, headers, etc.) from YOLO
    layout_map: Dict[str, Any]
    # The extracted text results from our Ensemble OCR
    raw_text: Annotated[List[str], operator.add]
    # Confidence scores for each extraction
    confidence_scores: Dict[str, float]
    # Final structured JSON output
    final_output: Dict[str, Any]