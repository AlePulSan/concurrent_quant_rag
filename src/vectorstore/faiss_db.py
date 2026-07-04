import faiss
import numpy as np
import logging
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)

class FAISSVectorStore:
    def __init__(self, dimension: int = 384):
        """
        Inicializa la base de datos vectorial en memoria
        Usa IndexFlatL2 para búsqueda con distancia euclidia
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        # Diccionario para mapear el ID interno de FAISS con el ID deltexto
        self.doc_map: Dict[int, str] = {}
        logger.info(f"FAISS inicializado con {dimension} dimensiones.")

    def add_vectors(self, embeddings: np.ndarray, chunk_ids: List[str]) -> None:
        """
        Inyecta los vectores en el índice
        """
        if len(embeddings) != len(chunk_ids):
            raise ValueError("El número de vectores no coincide con el número de IDs.")
        
        # faiis proporciona una matriz contigua
        vectores_float32 = np.array(embeddings).astype('float32')
        
        # Guardamos en qué índice numérico empieza a insertar FAISS
        start_idx = self.index.ntotal
        
        # Inserción en memortia
        self.index.add(vectores_float32)
        
        # Actualizamos nuestro mapa para no perder la referencia del texto
        for i, chunk_id in enumerate(chunk_ids):
            self.doc_map[start_idx + i] = chunk_id
            
        logger.debug(f"Insertados {len(chunk_ids)} vectores. Total en índice: {self.index.ntotal}")

    def search(self, query_vector: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
        """
        Busca los k vectores más cercanos a la pregunta.
        Retorna una lista de tuplas(ID_del_chunk, distancia_matemática)
        """
        if self.index.ntotal == 0:
            logger.warning("Intento de búsqueda en un índice FAISS vacío.")
            return []

        # Aseguramos que la pregunta también sea float32 y tenga dos dimensiones (1, dimension)
        q_vec = np.array(query_vector).astype('float32')
        if len(q_vec.shape) == 1:
            q_vec = q_vec.reshape(1, -1)

        # La búsqueda (c++)
        distancias, indices = self.index.search(q_vec, k)
        
        resultados = []
        for dist, idx in zip(distancias[0], indices[0]):
            # faiss devuelve -1 si no encuentra suficientes vecinos
            if idx != -1 and idx in self.doc_map:
                chunk_id = self.doc_map[idx]
                resultados.append((chunk_id, float(dist)))
                
        return resultados