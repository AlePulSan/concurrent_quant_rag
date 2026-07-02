import logging
from sentence_transformers import SentenceTransformer
from src.processing.workers import _naive_chunking
from src.vectorstore.faiss_db import FAISSVectorStore 
from src.ingestion.async_reader import AsyncDocumentReader
import aiohttp

logger = logging.getLogger(__name__)

class OrquestadorRAG:
    def __init__(self):
        # Se cargamos el modelo
        self.modelo = SentenceTransformer("all-MiniLM-L6-v2")
        # Inicializamos la base de datos vectorial en memoria
        self.db = FAISSVectorStore(dimension=384)
        self.lector = AsyncDocumentReader()
        self.memoria_textos = {}
        
    async def ingerir_lote_documentos(self, doc_ids: list[str]):
        """Flujo completo: Cortar -> Vectorizar -> Guardar en FAISS"""
        print(f"\nProcesando Documento: {doc_ids}")
        # Concurrencia
        documentos_crudos = await self.lector.read_batch(doc_ids)
        
        # procesado
        for doc in documentos_crudos:
            doc_id = doc["id"]
            texto = doc["texto"]
            
            if not texto:
                continue
                
            # Cortar
            chunks = _naive_chunking(texto, chunk_size=150, overlap=30)
            if not chunks:
                print("El documento está vacío.")
                continue
            
            print(f"Texto fragmentado en {len(chunks)} partes")
            
            # Generamos IDs únicos para cada trozo
            chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
            
            # Vectorizamos
            vectores = self.modelo.encode(chunks)
            
            # Guardamos
            self.db.add_vectors(vectores, chunk_ids)
        print("Vectores inyectados en la base de datos con éxito.")
        
    def buscar_respuesta(self, pregunta: str, k: int = 2):
        """Convierte la pregunta a vector, busca en FAISS y recupera el texto"""
        print(f"\nBuscando en la base de datos: '{pregunta}'")
        
        # Vectorizar la pregunta
        vector_pregunta = self.modelo.encode([pregunta])
        
        # Búsqueda
        resultados_faiss = self.db.search(vector_pregunta, k)
        
        contexto_recuperado = []
        for chunk_id, distancia in resultados_faiss:
            texto_real = self.memoria_textos.get(chunk_id, "")
            contexto_recuperado.append({"id": chunk_id, "texto": texto_real, "distancia": distancia})
            
        return contexto_recuperado

    async def generar_respuesta_llm(self, pregunta: str, modelo_ollama: str = "llama3.1"):
        """Ensambla el contexto y ataca la API de Ollama (local)"""
        # Recuperamos los mejores fragmentos
        documentos = self.buscar_respuesta(pregunta, k=3)
        bloque_contexto = "\n---\n".join([doc["texto"] for doc in documentos])
        
        # Prompt para definir cvontexto
        prompt_final = f"""You are a quantitative data engineer. Answer the question directly and objectively using ONLY the provided information. If the answer cannot be found in the text, state clearly that you do not have enough data.

CONTEXT INFORMATION:
{bloque_contexto}

USER QUESTION:
{pregunta}

TECHNICAL ANSWER:"""

        print(f"Lanzando inferencia a Ollama (Modelo: {modelo_ollama})...")
        
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": modelo_ollama,
            "prompt": prompt_final,
            "stream": False,
            "options": {"temperature": 0.0} 
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("response", "")
                    else:
                        return f"[ERROR API] Code {response.status}"
            except Exception as e:
                return f"[ERROR] Ollama communication failed: {e}"