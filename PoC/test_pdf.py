import sys
import asyncio
import time
from pathlib import Path

root_dir = Path(__file__).parent.parent.absolute()
sys.path.append(str(root_dir))

from src.orquestador import OrquestadorRAG

async def main():
    motor = OrquestadorRAG()
    archivo_pdf = "attention_paper.pdf" 
    
    # Ingesta
    await motor.ingerir_lote_documentos([archivo_pdf])
    
    # Pregunta
    pregunta = "What is a transformer and what main problem does it solve in sequence modeling?"
    print(f"\nQuestion: '{pregunta}'")
    
    # Inferencia
    inicio_inferencia = time.time()
    respuesta_final = await motor.generar_respuesta_llm(pregunta, "llama3.1")
    fin_inferencia = time.time()
    
    print("Answer:\n")
    print(respuesta_final)
    print(f"Inference latency: {fin_inferencia - inicio_inferencia:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())