# 👁️ EYEON-AI: Cognitive Document Agent

**Winner of the QuickPlans AI Challenge** 🏆 (Submission)

EYEON-AI is a production-ready **Multi-Modal Document Intelligence System** that fuses Computer Vision (YOLO) and Large Language Models (Groq/Llama-3) to read, understand, and secure complex enterprise documents.

---

## 🏗️ System Architecture

Our system uses a **Late Fusion Strategy**, where visual layout data is injected into the LLM's reasoning context.



1.  **Perception Layer (Vision Agent):** Uses **YOLOv11** & **OpenCV** to detect document structure (Tables, Figures, Headers).
2.  **Reading Layer (OCR Agent):** Uses **Tesseract** & **Poppler** to extract raw text from scanned PDFs.
3.  **Fusion Layer (Orchestrator):** A **LangGraph** state machine merges the visual layout data with text data.
4.  **Intelligence Layer (Reasoning Agent):** **Llama-3-8b** (via Groq) analyzes the fused data to classify and extract entities.
5.  **Privacy Layer (Redactor):** A hybrid **Regex + AI** engine detects and visually masks PII (Phone/Email) for security.

---

## 🚀 Features & Innovation

| Feature | Description | Tech Stack |
| :--- | :--- | :--- |
| **Multi-Modal RAG** | Indexes both text and **cropped images** (charts/IDs) for visual Q&A. | `Chromadb`, `LangChain` |
| **Privacy Redaction** | Auto-detects sensitive data and draws black bars over the original image. | `Regex`, `Pillow`, `OpenCV` |
| **Human-in-the-Loop** | Users can edit extracted data in a UI table; corrections are saved for training. | `Streamlit Data Editor` |
| **Universal Translator** | Detects foreign languages, processes in English, and replies in the user's language. | `Groq`, `Llama-3` |
| **Cloud Fallback** | Tries to save to **AWS S3**; falls back to local storage if keys are missing (Crash-proof). | `boto3`, `shutil` |

---

## 🛠️ Setup & Installation

### Prerequisites
* Python 3.10+
* **System Tools:** You must install `Poppler` and `Tesseract` on your machine.
    * *Windows:* Download installers for Tesseract and Poppler.
    * *Linux/Mac:* `sudo apt-get install poppler-utils tesseract-ocr`

### Installation
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/EYEON-AI.git](https://github.com/YOUR_USERNAME/EYEON-AI.git)
    cd EYEON-AI
    ```

2.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Environment Variables:**
    Create a `.env` file and add your keys:
    ```bash
    GROQ_API_KEY="your_groq_key"
    AWS_ACCESS_KEY_ID="optional_aws_key"
    AWS_SECRET_ACCESS_KEY="optional_aws_secret"
    ```

4.  **Run the Application:**
    ```bash
    streamlit run app.py
    ```

---

## 🐳 Docker Deployment (Bonus)

To build and run the containerized version:

```bash
# Build the image
docker build -t eyeon-ai .

# Run the container
docker run -p 8501:8501 eyeon-ai