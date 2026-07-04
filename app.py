import streamlit as st
import asyncio
import os
from src.orquestador import OrquestadorRAG
from src.config import RAGConfig

st.set_page_config(page_title="Quant RAG", page_icon="⚙️", layout="wide")
st.title("Concurrent Quant RAG Engine")

# Inicialización de estado y memoria
if "motor" not in st.session_state:
    st.session_state.motor = OrquestadorRAG()
    st.session_state.ingestado = False

# Aquí inicializamos la memoria del chat en el frontend
if "messages" not in st.session_state:
    st.session_state.messages = []

# Config del panel lateralr
with st.sidebar:
    st.header("Data Ingestion")
    pdf_name = st.text_input("Filename:", "attention_paper.pdf")
    
    if st.button("Run Vectorization"):
        # purgamos el historial de chat e iniciamos un nuevo motor
        st.session_state.motor = OrquestadorRAG()
        st.session_state.ingestado = False
        st.session_state.messages = [] 
        
        ruta_absoluta = os.path.abspath(os.path.join(RAGConfig.DATA_RAW_DIR, pdf_name))
        
        if not os.path.exists(ruta_absoluta):
            st.error(f"Archivo no encontrado en {ruta_absoluta}")
        else:            
            with st.spinner("Purging memory & Indexing from zero..."):
                asyncio.run(st.session_state.motor.ingerir_lote_documentos([pdf_name]))
                st.session_state.ingestado = True
            st.success("Indexing Complete!")

st.markdown("---")

# Renderizado del historial de chat
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Lógica
pregunta = st.chat_input("Enter your technical query...")

if pregunta:
    if not st.session_state.ingestado:
        st.error("Access Denied: Index a document first.")
    else:
        # Añadir y renderizar la pregunta
        st.chat_message("user").write(pregunta)
        st.session_state.messages.append({"role": "user", "content": pregunta})
        
        with st.spinner("LLM is processing..."):
            # Inyectamos el historial completo al backend (el backend ya lo recorta con la sliding window)
            respuesta = asyncio.run(
                st.session_state.motor.generar_respuesta_llm(
                    pregunta=pregunta,
                    historial_chat=st.session_state.messages
                )
            )
            
        # respuesta del sistema
        st.chat_message("assistant").write(respuesta)
        st.session_state.messages.append({"role": "assistant", "content": respuesta})
        
        # 6. Observabilidad (Debugging)
        with st.expander("Ver Contexto (FAISS Retrieval)"):
            contexto_crudo = st.session_state.motor.buscar_respuesta(pregunta, k=RAGConfig.RETRIEVAL_K)
            
            for i, doc in enumerate(contexto_crudo):
                try:
                    distancia = doc.get('distancia', 'N/A') if isinstance(doc, dict) else getattr(doc, 'distancia', 'N/A')
                    
                    if isinstance(doc, dict):
                        texto_chunk = doc.get('texto', doc.get('text', doc.get('page_content', str(doc))))
                    else:
                        texto_chunk = getattr(doc, 'page_content', getattr(doc, 'text', str(doc)))
                        
                    st.markdown(f"**Chunk {i+1} (Distancia: {distancia}):**")
                    st.text(texto_chunk)
                    
                except Exception as e:
                    st.error(f"Error parseando la estructura de datos del chunk {i+1}: {str(e)}")
                    st.write(doc)