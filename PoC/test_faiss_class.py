import sys
from pathlib import Path
import numpy as np

root_dir = Path(__file__).parent.parent.absolute()
sys.path.append(str(root_dir))

# Importamos la clase que acabas de crear en producción
from src.vectorstore.faiss_db import FAISSVectorStore

def main():    
    # 1. Instanciamos nuestra clase
    db = FAISSVectorStore(dimension=3)
    
    # 2. Simulamos 4 vectores (embeddings de juguete de 3 dimensiones)
    vectores_simulados = np.array([
        [0.1, 0.2, 0.3],
        [0.9, 0.8, 0.7],
        [0.2, 0.1, 0.4],
        [0.8, 0.9, 0.6]
    ])
    
    # 3. Simulamos los IDs de esos textos
    ids_textos = ["doc_1_chunk_1", "doc_2_chunk_1", "doc_1_chunk_2", "doc_2_chunk_2"]
    
    # 4. Insertamos en el motor
    print("Insertando vectores en FAISS...")
    db.add_vectors(vectores_simulados, ids_textos)
    
    # 5. Simulamos una pregunta (vector cercano al primer grupo)
    pregunta = np.array([[0.1, 0.2, 0.25]])
    print("\nBuscando vectores similares...")
    
    resultados = db.search(pregunta, k=2)
    
    for chunk_id, distancia in resultados:
        print(f"Encontrado: {chunk_id} | Distancia: {distancia:.4f}")

if __name__ == "__main__":
    main()