from typing import List

def _naive_chunking(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Divide un texto masivo en fragmentos (chunks) con solapamiento.
    Uso de slicing nativo para máxima eficiencia en memoria.
    """
    if not text:
        return []
    
    # Dividimos el texto por espacios en blanco para hacerlo de manera simplificada
    # Normalmente se usan expresiones regulares para no cortar las frases
    palabras = text.split()
    chunks = []
    
    if overlap >= chunk_size:
        raise ValueError("El solapamiento debe ser menor que el tamaño del chunk.")
        
    salto = chunk_size - overlap 
    
    for i in range(0, len(palabras), salto):
        # Slicing hiper-rápido en C por debajo
        bloque = palabras[i : i + chunk_size]
        
        # Volvemos a unir el array en texto plano para el modelo de IA
        chunks.append(" ".join(bloque)) 
        
        # Optimización: Si el índice actual más el tamaño supera el total, 
        # significa que ya hemos agarrado el final del texto. Rompemos el bucle.
        if i + chunk_size >= len(palabras):
            break
            
    return chunks