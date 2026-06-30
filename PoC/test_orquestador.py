import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.absolute()
sys.path.append(str(root_dir))
from src.orquestador import OrquestadorRAG

def main():    
    orquestador = OrquestadorRAG()
    
    # Texto de ejemplo con diversos temas
    texto_crudo = (
        "El mercado de semiconductores cerró al alza impulsado por la IA. "
        "Por otro lado, el sistema de transmisión DJI O4 Pro montado en un chasis SpeedyBee Mario Mini 25 "
        "reduce drásticamente la latencia de vídeo en vuelos acrobáticos, superando a las generaciones anteriores. "
        "Mientras tanto, los tipos de interés se mantienen estables en Europa."
    )
    
    # Usamos el orquestador para transformar y guardar en la base de datos vectoriual
    orquestador.ingerir_documento(texto_crudo, doc_id="Prueba_Final_Martes")
    
    # Hacemos una pregunta específica
    pregunta = "¿Qué sistema reduce la latencia en el chasis Mario Mini 25?"
    resultados = orquestador.buscar_respuesta(pregunta, k=1)
    
    print("\n Resultado")
    for chunk_id, distancia in resultados:
        print(f"ID Encontrado: {chunk_id}")
        print(f"Distancia Matemática: {distancia:.4f}")
        if distancia < 1.0:
            print("[Success]")
        else:
            print("[Err] La distancia es muy alta")

if __name__ == "__main__":
    main()