from pdf2image import convert_from_path
import platform

def pdf_to_images(pdf_path, dpi=300, poppler_path=None):
    """
    Converts PDF pages to a list of PIL images.
    Auto-detects Cloud vs Windows environment.
    """
    
    # TRICK: If we are on Linux (Streamlit Cloud), IGNORE the Windows path!
    if platform.system() == "Linux":
        poppler_path = None  
    
    # If we are on Windows, keep using the path you passed in.
    
    return convert_from_path(
        pdf_path, 
        dpi=dpi, 
        poppler_path=poppler_path
    )
