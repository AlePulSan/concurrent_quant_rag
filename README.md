# Concurrent Quant RAG

Async RAG engine for local, concurrent ingestion and analysis of technical PDFs. Optimized for 100% local inference (Full VRAM Offload) and non-blocking I/O.

## Tech Stack
* Python 3.x
* PyMuPDF (Async Extraction)
* Sentence-Transformers (Embeddings)
* FAISS (Vector DB)
* aiohttp & Ollama (Local LLM Inference)

## Setup & Run

1. Install dependencies:
pip install -r requirements.txt

2. Start the local LLM (Ollama required):
ollama run llama3.1

3. Add Data:
Place your technical `.pdf` files inside the `data/raw/` folder.

4. Run the Pipeline:
python PoC/test_real_pdf.py