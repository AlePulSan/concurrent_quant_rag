import streamlit as st
import asyncio
import os
from src.orquestador import OrquestadorRAG
from src.config import RAGConfig

st.set_page_config(page_title="Quant RAG", layout="wide")
st.title("Concurrent Quant RAG Engine")

# Inicialización de estado y memoria del motor
if "motor" not in st.session_state:
    st.session_state.motor = OrquestadorRAG()
    st.session_state.ingestado = False

# Inicializamos la memoria del chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# config del panel lateral izq 
with st.sidebar:
    st.header("Data")
    
    # Uploader de Streamlit
    archivo_subido = st.file_uploader("Sube el informe 10-K (PDF)", type=["pdf"])
    
    if archivo_subido is not None:
        # Guardamos el archivo en la carpeta raw para que el AsyncReader lo encuentre
        ruta_absoluta = os.path.abspath(os.path.join(RAGConfig.DATA_RAW_DIR, archivo_subido.name))
        
        # Volcamos el buffer a disco
        with open(ruta_absoluta, "wb") as f:
            f.write(archivo_subido.getbuffer())
            
        st.success(f"Archivo alojado en servidor: {archivo_subido.name}")
        
        # El botón solo aparece si hay un archivo subido
        if st.button("Run Vectorization"):
            # Hard reset de memoria y motor vectorial para evitar contaminación
            st.session_state.motor = OrquestadorRAG()
            st.session_state.ingestado = False
            st.session_state.messages = [] 
            # Limpiamos caché previa
            if "extracted_data" in st.session_state:
                del st.session_state["extracted_data"]
            
            with st.spinner("Indexing from zero..."):
                asyncio.run(st.session_state.motor.ingerir_lote_documentos([archivo_subido.name]))
                st.session_state.ingestado = True
            st.success("Indexing Complete! Motor vectorial y memoria listos.")

st.markdown("---")

# dashboard
if st.session_state.ingestado:
    st.subheader("Company Snapshot")
    
    # Solo extraemos si no tenemos datos en caché
    if "extracted_data" not in st.session_state:
        with st.spinner("Extrayendo métricas fundamentales del 10-K..."):
            st.session_state.extracted_data = asyncio.run(st.session_state.motor.extraer_fundamentales_json())

    # Usamos .get() por seguridad para evitar KeyErrors si el LLM falla
    datos = st.session_state.extracted_data or {}

    if not datos:
        st.warning("No se pudo construir el dashboard con los datos. Se muestran valores por defecto.")
        
    # Extraemos OCF y CapEx, forzando a 0 si fallan
    ocf_curr = datos.get("ocf_current", 0)
    capex_curr = datos.get("capex_current", 0)
    ocf_prev = datos.get("ocf_prev", 0)
    capex_prev = datos.get("capex_prev", 0)
    
    fcf_current = ocf_curr - capex_curr
    fcf_prev = ocf_prev - capex_prev
    delta_fcf = fcf_current - fcf_prev
    
    # Titular dinámico con empresa y año
    company_name = datos.get("company", "Target Company")
    year_current = datos.get("year_current", "N/A")
    st.markdown(f"### 🏢 {company_name} (FY {year_current})")

    # KPIs con colores (Deltas)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Free Cash Flow (M$)", 
                  value=f"${fcf_current}", 
                  delta=f"{delta_fcf} YoY", delta_color="normal")
        
    with col2:
        # ROIC
        curr_roic = datos.get("roic_current", 0)
        st.metric(label="ROIC (%)", 
                  value=f"{curr_roic}%", 
                  delta="Model Output", delta_color="off")
        
    with col3:
        # Deuda: Más bajo es mejor (inverse)
        curr_debt = datos.get("short_term_debt", 0)
        st.metric(label="Short-Term Debt (M$)", 
                  value=f"${curr_debt}", 
                  delta="Model Output", delta_color="off")

    st.divider()

    # Gráfico
    st.markdown("**Evolución del Free Cash Flow (M$)**")
    
    tendencia_ocf = datos.get("trend_ocf", {})
    tendencia_capex = datos.get("trend_capex", {})
    
    if tendencia_ocf and tendencia_capex:
        # Calculamos el FCF año a año dinámicamente
        fcf_trend = {}
        for year in tendencia_ocf.keys():
            # Aseguramos que los valores sean flotantes
            try:
                val_ocf = float(tendencia_ocf.get(year, 0))
                val_capex = float(tendencia_capex.get(year, 0))
                fcf_trend[year] = val_ocf - val_capex
            except (ValueError, TypeError):
                fcf_trend[year] = 0
                
        st.bar_chart(fcf_trend, height=200, color="#1f77b4")
    else:
        st.info("Datos de tendencia insuficientes en el modelo de extracción para hacer el grafico")
    
    st.markdown("---")

# Renderizado del historial de chat
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Lógica del Chat
pregunta = st.chat_input("Enter your technical query...")

if pregunta:
    if not st.session_state.ingestado:
        st.error("Access Denied: Index a document first.")
    else:
        # Añadir y renderizar la pregunta
        st.chat_message("user").write(pregunta)
        st.session_state.messages.append({"role": "user", "content": pregunta})
        
        with st.spinner("LLM is processing..."):
            # Inyectamos el historial completo y el contexto cuantitativo al backend
            contexto_cuantitativo = str(st.session_state.get("extracted_data", {}))
            respuesta = asyncio.run(
                st.session_state.motor.generar_respuesta_llm(
                    pregunta=pregunta,
                    historial_chat=st.session_state.messages,
                    contexto_cuantitativo=contexto_cuantitativo
                )
            )
            
        # respuesta del sistema
        st.chat_message("assistant").write(respuesta)
        st.session_state.messages.append({"role": "assistant", "content": respuesta})
        
        # Observabilidad para debugging
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