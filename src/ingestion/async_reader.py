import asyncio
import logging
import time
import pathlib
from typing import List, Dict
import fitz

logger = logging.getLogger(__name__)

class AsyncDocumentReader:
    def __init__(self, base_dir: str = "data/raw"):
        # Usa el diccionario que pasemos como almacenamiento. Si no se usa uno vacío.
        self.base_dir = pathlib.Path(base_dir)

    def _read_pdf_sync(self, file_path: pathlib.Path) -> str:
        """
        Extrae texto de un pdf
        """
        texto = []
        try:
            with fitz.open(file_path) as doc:
                for page in doc:
                    texto.append(page.get_text())
            return "\n".join(texto)
        except Exception as e:
            logger.error(f"Error al leer el archivo {file_path.name}: {e}")
            return ""
        
    async def _read_single_document(self, doc_id: str) -> Dict[str, str]:
        """
        lectura de un archivo
        """        
        file_path = self.base_dir / doc_id

        if not file_path.exists():
            print(f" Error: '{doc_id}' no existe en la carpeta '{self.base_dir}'")

        texto = await asyncio.to_thread(self._read_pdf_sync, file_path)
        
        print(f"{doc_id}: Extracción completada. ({len(texto)} caracteres)")
        return {"id": doc_id, "texto": texto}


    async def read_batch(self, doc_ids: List[str]) -> List[Dict[str, str]]:
        """
        Dispara la lectura de todos los documentos en paralelo (Concurrencia).
        """
        tareas = [self._read_single_document(doc_id) for doc_id in doc_ids]
        resultados = await asyncio.gather(*tareas)
        return resultados