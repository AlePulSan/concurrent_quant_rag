import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.absolute()
sys.path.append(str(root_dir))
from src.processing.workers import _naive_chunking

def main():
    texto= (
        "Los mercados financieros actuales operan con cautela ante el debate sobre si el reciente repunte de la bolsa está justificado o si anticipa una nueva burbuja,"
        " fuertemente ligada al sector de la inteligencia artificial. A nivel internacional, Wall Street registra caídas semanales consecutivas lideradas por el sector tecnológico," 
        "mientras que las materias primas sufren correcciones notables, con el barril de petróleo cayendo por debajo de los 70 dólares."
    )
    
    # Cortamos el texto en trozos de 15 palabras, solapando 5
    chunks = _naive_chunking(text=texto, chunk_size=15, overlap=5)
    
    print(f"\nCantidad de Chunks generados: {len(chunks)}\n")
    
    for i, chunk in enumerate(chunks):
        print(f"[Fragmento {i}]: {chunk}")

    

if __name__ == "__main__":
    main()