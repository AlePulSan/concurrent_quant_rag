from typing import List

def _naive_chunking(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Divide un texto en fragmentos (chunks) añadiendo solapamiento.
    """
    if not text:
        return []
    
    # Dividimos el texto por espacios en blanco para hacerlo de manera simplificada
    # Normalmente se usan expresiones regulares para no cortar frases
    palabras = text.split()
    chunks = []
    
    if overlap >= chunk_size:
        raise ValueError("El solapamiento debe ser menor que el tamaño del chunk")
        
    salto = chunk_size - overlap 
    
    for i in range(0, len(palabras), salto):
        # Slicing
        bloque = palabras[i : i + chunk_size]
        
        # Volvemos a unir el array en texto plano para el modelo
        chunks.append(" ".join(bloque)) 
        if i + chunk_size >= len(palabras):
            break
            
    return chunks