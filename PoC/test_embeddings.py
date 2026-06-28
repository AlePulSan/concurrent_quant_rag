from sentence_transformers import SentenceTransformer

def main():   
    # cargamos el modelo para la transformación en memoria
    modelo = SentenceTransformer("all-MiniLM-L6-v2")
    
    texto = "Los mercados financieros actuales operan con cautela"
    
    print(f"\nTexto de entrada: '{texto}'")
    print("Calculando coordenadas...")
    
    # Texto -> Coordenadas
    vector = modelo.encode(texto)

    print(f"Tipo de estructura: {type(vector)}")
    print(f"Dimensiones del vector (Shape): {vector.shape}")
    
    # Imprimimos las 5 primeras coordenadas
    print(f"Primeras 5 coordenadas: {vector[:5]}")

if __name__ == "__main__":
    main()