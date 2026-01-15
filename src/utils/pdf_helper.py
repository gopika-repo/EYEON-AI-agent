from pdf2image import convert_from_path

def pdf_to_images(pdf_path, dpi=300, poppler_path=None):
    """
    Converts PDF pages to a list of PIL images.
    Updated for 2026 Windows stability with explicit Poppler pathing.
    """
    # By adding poppler_path here, we bridge the gap between Python and Windows
    return convert_from_path(
        pdf_path, 
        dpi=dpi, 
        poppler_path=poppler_path
    )