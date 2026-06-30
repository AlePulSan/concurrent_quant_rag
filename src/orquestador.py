import logging
from sentence_transformers import SentenceTransformer
from src.processing.workers import _naive_chunking
from src.vectorstore.faiss_db import FAISSVectorStore 

logger = logging.getLogger(__name__)

class OrquestadorRAG:
    def __init__(self):
        # Se cargamos el modelo
        self.modelo = SentenceTransformer("all-MiniLM-L6-v2")
        # Inicializamos la base de datos vectorial en memoria
        self.db = FAISSVectorStore(dimension=384)
        
    def ingerir_documento(self, texto_crudo: str, doc_id: str):
        """Flujo completo: Cortar -> Vectorizar -> Guardar en FAISS"""
        print(f"\nProcesando Documento: {doc_id}")
        
        # Cortar
        chunks = _naive_chunking(texto_crudo, chunk_size=20, overlap=5)
        if not chunks:
            print("El documento está vacío.")
            return
        
        print(f"Texto fragmentado en {len(chunks)} partes")
        
        # Generamos IDs únicos para cada trozo
        chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        
        # Vectorizamos
        vectores = self.modelo.encode(chunks)
        
        # Guardamos
        self.db.add_vectors(vectores, chunk_ids)
        print("Vectores inyectados en la base de datos con éxito.")
        
    def buscar_respuesta(self, pregunta: str, k: int = 2):
        """Convierte la pregunta a vector y busca en FAISS"""
        print(f"\nBuscando en la base de datos: '{pregunta}'")
        
        # Vectorizar la pregunta
        vector_pregunta = self.modelo.encode([pregunta])
        
        # Búsqueda ultrarrápida
        resultados = self.db.search(vector_pregunta, k)
        return resultados