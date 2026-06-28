import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def main():
    modelo = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Simulamos datos
    base_de_datos = [
        "Nvidia presenta sus nuevas GPUs para centros de datos.",
        "AMD gana cuota de mercado en procesadores de servidor.",
        "La inflación en Estados Unidos afecta a los tipos de interés.",
        "TSMC advierte sobre problemas en la cadena de suministro en Taiwán.",
        "La receta de la tortilla de patatas lleva huevos y patatas."
    ]
    
    print("Vectorizando la base de datos")
    vectores_db = modelo.encode(base_de_datos)
    
    print("\n CREANDO ÍNDICE FAISS EN C++")
    dimensiones = vectores_db.shape[1] # Esto será 384
    
    # IndexFlatL2 es el índice más rudo
    # mide la distancia Euclidiana (L2) en el espacio
    indice = faiss.IndexFlatL2(dimensiones)
    
    # FAISS exige matrices contiguas en memoria y formato float32 estricto
    vectores_db = np.array(vectores_db).astype('float32')
    
    # Cargamos los vectores en la memoria del índice
    indice.add(vectores_db)
    print(f"Vectores indexados y listos en memoria: {indice.ntotal}")
    
    print("\n Se simula una pregunta")
    pregunta = "¿Qué está pasando con la fabricación de chips en Asia?"
    print(f"Pregunta cruda: '{pregunta}'")
    
    # La pregunta también tiene que mutar al mismo espacio dimensional de 384 coordenadas
    vector_pregunta = modelo.encode([pregunta])
    vector_pregunta = np.array(vector_pregunta).astype('float32')
    
    print("\n Busqueda")
    # k=2 significa: "Búscame los 2 fragmentos con menor distancia matemática a la pregunta"
    k = 2 
    distancias, indices_resultados = indice.search(vector_pregunta, k)
    
    for i in range(k):
        # FAISS nos devuelve el ID (la posición) del vector ganador y su distancia
        # Al ser un modelo y unos datos trmendamente sencillos se ve compromentida la exactitud de la respuesta 
        id_resultado = indices_resultados[0][i]
        distancia = distancias[0][i]
        texto_encontrado = base_de_datos[id_resultado]
        
        print(f"[Top {i+1}] (Distancia Euclidiana: {distancia:.2f}) -> {texto_encontrado}")

if __name__ == "__main__":
    main()