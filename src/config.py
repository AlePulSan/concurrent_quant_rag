import os

class RAGConfig:
    '''Parametros generales del proyecto'''
    # Rutas
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
    
    # Modelo
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    VECTOR_DIMENSION = 384
    
    # Procesamiento del texto
    CHUNK_SIZE = 150
    CHUNK_OVERLAP = 30
    
    # FAISS
    RETRIEVAL_K = 6 # Número de fragmentos a recuperar en la búsqueda
    
    # LLM (Ollama)
    OLLAMA_URL = "http://localhost:11434/api/generate"
    DEFAULT_LLM_MODEL = "llama3.1"
    LLM_TEMPERATURE = 0.0
    
    # Memoria tope del chat
    MEMORY_WINDOW_SIZE = 3 # Número de interacciones (pregunta/respuesta) que el bot recordará