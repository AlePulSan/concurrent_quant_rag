from sentence_transformers import SentenceTransformer
from src.processing.workers import _naive_chunking
from src.vectorstore.faiss_db import FAISSVectorStore 
from src.ingestion.async_reader import AsyncDocumentReader
from src.config import RAGConfig
import logging
import aiohttp
import pymupdf4llm
import os
import json
import re

logger = logging.getLogger(__name__)

class OrquestadorRAG:
    def __init__(self):
        self.modelo = SentenceTransformer(RAGConfig.EMBEDDING_MODEL_NAME)
        self.db = FAISSVectorStore(dimension=RAGConfig.VECTOR_DIMENSION)
        self.lector = AsyncDocumentReader()
        self.memoria_textos = {}
        
    async def ingerir_lote_documentos(self, doc_ids: list[str]):
        """Flujo (Cortar -> Vectorizar -> Guardar en FAISS)"""
        print(f"\nProcesando: {doc_ids}")
        documentos = await self.lector.read_batch(doc_ids)
        
        for doc in documentos:
            doc_id = doc["id"]
            texto = doc["texto"]
            
            if not texto:
                continue
                
            # Chunking (cortar)
            chunks = _naive_chunking(texto, chunk_size=RAGConfig.CHUNK_SIZE, overlap=RAGConfig.CHUNK_OVERLAP)

            if not chunks:
                print("El documento está vacío.")
                continue
            
            print(f"Texto fragmentado en {len(chunks)} partes")
            chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
            
            # Vectorizamos y Guardamos
            vectores = self.modelo.encode(chunks)
            self.db.add_vectors(vectores, chunk_ids)
            
            # Se guarda en memoria para poder recuperar el texto después
            for i, chunk_id in enumerate(chunk_ids):
                self.memoria_textos[chunk_id] = chunks[i]
                
        print("Vectores inyectados en la base de datos")
        
    def buscar_respuesta(self, pregunta: str, k: int = RAGConfig.RETRIEVAL_K):
        """Convierte la pregunta a vector, busca en FAISS y recupera el contexto"""
        print(f"\nBuscando en la base de datos: '{pregunta}'")
        
        vector_pregunta = self.modelo.encode([pregunta])
        resultados_faiss = self.db.search(vector_pregunta, k)
        
        contexto_recuperado = []
        for chunk_id, distancia in resultados_faiss:
            texto_real = self.memoria_textos.get(chunk_id, "")
            contexto_recuperado.append({"id": chunk_id, "texto": texto_real, "distancia": distancia})
            
        return contexto_recuperado

    async def generar_respuesta_llm(self, pregunta: str, historial_chat: list = None, contexto_cuantitativo: str = None, modelo_ollama: str = RAGConfig.DEFAULT_LLM_MODEL):
        """Ensambla el contexto, la memoria reciente y ataca la API local"""
        # Recuperamos contexto
        documentos = self.buscar_respuesta(pregunta, k=RAGConfig.RETRIEVAL_K)
        bloque_contexto = "\n---\n".join([doc["texto"] for doc in documentos])
        
        # Añadimos los datos basicos extraídos
        if contexto_cuantitativo:
            bloque_contexto += f"\n\n[QUANTITATIVE DATA - FUNDAMENTALS]:\n{contexto_cuantitativo}\n---\n"
        
        # Implementamos memoria del chat
        bloque_memoria = ""
        if historial_chat:
            mensajes_recientes = historial_chat[-(RAGConfig.MEMORY_WINDOW_SIZE * 2):] # <- Se multiplica por dos porque cada interacción conlleva una pregunta y una respuesta
            for msg in mensajes_recientes:
                rol = "USER" if msg["role"] == "user" else "ASSISTANT"
                bloque_memoria += f"{rol}: {msg['content']}\n"
        else:
            bloque_memoria = "No previous conversation history."
        
        # Prompt
        prompt_final = f"""You are a strict quantitative data engineer. Answer the question directly using ONLY the provided CONTEXT INFORMATION. 

CRITICAL RULE: If the answer is not explicitly found in the CONTEXT INFORMATION, you must reply with exactly "DATA NOT FOUND IN CONTEXT" and nothing else. Do not use external knowledge. Do not apologize.

PREVIOUS CONVERSATION HISTORY (Use to understand pronouns or follow-up questions):
{bloque_memoria}

CONTEXT INFORMATION:
{bloque_contexto}

USER QUESTION:
{pregunta}

TECHNICAL ANSWER:"""

        print(f"Lanzando inferencia a Ollama (Modelo: {modelo_ollama})...")
        
        payload = {
            "model": modelo_ollama,
            "prompt": prompt_final,
            "stream": False,
            "options": {"temperature": RAGConfig.LLM_TEMPERATURE} 
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(RAGConfig.OLLAMA_URL, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("response", "")
                    else:
                        return f"[ERROR API] Code {response.status}"
            except Exception as e:
                return f"[ERROR] Ollama communication failed: {e}"
    
    async def extraer_fundamentales_json(self):
        """Fuerza al LLM a devolver un JSON estricto con métricas contables puras"""
        
        # Búsqueda adaptada a la nomenclatura oficial de la SEC
        documentos = self.buscar_respuesta("Net cash provided by operating activities, Purchases of property and equipment, Current portion of long-term debt, Commercial paper, ROIC", k=10)
        bloque_contexto = "\n---\n".join([doc["texto"] for doc in documentos])
        
        prompt_json = f"""You are a strict quantitative data extraction engine.
Analyze the CONTEXT INFORMATION and extract the specific accounting lines.
Return ONLY a valid, parseable JSON object. No explanations.

REQUIRED JSON FORMAT:
{{
    "company": "<Name of the company>",
    "year_current": "<Most recent fiscal year found (e.g., '2022')>",
    "year_prev": "<Previous fiscal year found (e.g., '2021')>",
    "ocf_current": <number, "Net cash provided by operating activities" for the most recent year>,
    "ocf_prev": <number, "Net cash provided by operating activities" for the previous year>,
    "capex_current": <number, "Purchases of property and equipment" for the most recent year. MUST BE POSITIVE>,
    "capex_prev": <number, "Purchases of property and equipment" for the previous year. MUST BE POSITIVE>,
    "short_term_debt": <number, "Current portion of long-term debt" plus "Commercial paper". Do NOT use Total Current Liabilities. Use 0 if not found>,
    "roic_current": <number, Return on Invested Capital. Use 0 if not explicitly found>,
    "trend_ocf": {{
        "<Year N-2>": <number>,
        "<Year N-1>": <number>,
        "<Year N>": <number>
    }},
    "trend_capex": {{
        "<Year N-2>": <number, positive>,
        "<Year N-1>": <number, positive>,
        "<Year N>": <number, positive>
    }}
}}

CONTEXT INFORMATION:
{bloque_contexto}
"""
        
        payload = {
            "model": RAGConfig.DEFAULT_LLM_MODEL,
            "prompt": prompt_json,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0}
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(RAGConfig.OLLAMA_URL, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        respuesta_cruda = data.get("response", "")
                        # Limpiamos por si el LLM mete un tag de markdown por error
                        respuesta_limpia = re.sub(r"```json|```", "", respuesta_cruda).strip()
                        return json.loads(respuesta_limpia)
                    else:
                        return None
            except Exception as e:
                print(f"Error extrayendo JSON: {e}")
                return None