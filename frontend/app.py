# frontend/app.py
import streamlit as st
import sys
import os
from datetime import datetime, timedelta
import json
import hashlib
import pandas as pd
from io import BytesIO
import base64

# Librerías para PDF
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Añadir backend al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
from analizador_optimizado import DataIntegrator, AgenteAnaliticoLLM

# ========================================
# CONFIGURACIÓN BÁSICA
# ========================================
st.set_page_config(
    page_title="Análista IA PAPE-AO",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #4d0f0e 0%, #341616 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f5f0ed;
        border-left: 4px solid #4d0f0e;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #ffe5e5;
        border-left: 4px solid #d32f2f;
        margin: 1rem 0;
    }
    .kpi-card {
        padding: 1.5rem;
        border-radius: 0.75rem;
        background: linear-gradient(135deg, #4d0f0e15 0%, #34161615 100%);
        border: 1px solid #4d0f0e30;
        text-align: center;
    }
    .chat-container {
        max-width: 900px;
        margin: 0 auto;
    }
    [data-testid="stMetricValue"] {
        color: #4d0f0e;
    }
    button[kind="primary"] {
        background-color: #4d0f0e !important;
    }
    button[kind="primary"]:hover {
        background-color: #341616 !important;
    }
    [data-testid="stSlider"] {
        accent-color: #4d0f0e;
    }
    </style>
""", unsafe_allow_html=True)

# ========================================
# FUNCIONES AUXILIARES
# ========================================

def hash_password(password):
    """Hashea una contraseña"""
    return hashlib.sha256(password.encode()).hexdigest()

def verificar_credenciales(usuario, contraseña):
    """Verifica credenciales contra los secrets de Streamlit"""
    try:
        usuarios = st.secrets["USUARIOS"]
        st.write("🔍 **DEBUG:** Usuarios cargados:", usuarios)  # ← Línea de debug
        
        if usuario in usuarios:
            hash_input = hash_password(contraseña)
            hash_almacenado = usuarios[usuario]
            st.write(f"🔑 Hash ingresado: {hash_input}")
            st.write(f"💾 Hash almacenado: {hash_almacenado}")
            st.write(f"✅ ¿Coinciden? {hash_input == hash_almacenado}")
            return hash_input == hash_almacenado
    except Exception as e:
        st.error(f"❌ Error cargando credenciales: {e}")
    return False

def obtener_limite_consultas():
    """Obtiene el límite de consultas del usuario"""
    try:
        return st.secrets.get("LIMITE_CONSULTAS_DIARIAS", 50)
    except:
        return 50

def verificar_limite_consultas():
    """Verifica si el usuario ha alcanzado su límite diario"""
    if st.session_state.usuario == "demo":
        return True, None
    
    limite = obtener_limite_consultas()
    consultas_hoy = st.session_state.consultas_realizadas
    
    return consultas_hoy < limite, limite - consultas_hoy

def generar_sugerencias(prompt_anterior):
    """Genera sugerencias inteligentes basadas en la pregunta anterior"""
    sugerencias = {
        "Población": [
            "¿Cuál es la población en Lomas de Becerra?",
            "¿Cuántos niños de 0 a 3 años?"
        ],
        "Mujeres": [
            "Personas elegibles para Pensión Mujeres Bienestar",
            "Mujeres con carencia de salud"
        ],
        "Carencia Salud": [
            "¿Personas con carencia a salud?",
            "Personas elegibles a IMSS Bienestar en Santa Lucía"
        ],
        "Tabla cruzada": [
            "Tabla cruzada por sexo y carencia seguridad social"
        ],
        "AGEB": [
            "Población total por AGEB",
            "Población con rezago educativo en AGEB 10370"
        ]
    }
    
    for keyword, sugg_list in sugerencias.items():
        if keyword.lower() in prompt_anterior.lower():
            return sugg_list
    
    return [
        "¿Población por colonia",
        "Personas elegibles Rita Cetina"
    ]

def crear_pdf_reporte(conversacion, usuario):
    """Crea un PDF con la conversación"""
    if not REPORTLAB_AVAILABLE:
        return None
    
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Título
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#4d0f0e'),
            spaceAfter=30,
            alignment=1
        )
        story.append(Paragraph("Análisis AI PAPE-AO", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Metadatos
        meta_style = ParagraphStyle(
            'Meta',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.grey,
        )
        story.append(Paragraph(f"Usuario: {usuario} | Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}", meta_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Conversación
        for msg in conversacion:
            role = "👤 Usuario" if msg["role"] == "user" else "🤖 Asistente"
            story.append(Paragraph(f"<b>{role}</b>", styles['Heading3']))
            story.append(Paragraph(msg["content"], styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
        
        doc.build(story)
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"Error generando PDF: {str(e)}")
        return None

def mostrar_pantalla_login():
    """Muestra la pantalla de login"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<h1 class="main-header">Análisis AI PAPE-AO</h1>', unsafe_allow_html=True)
        st.markdown("*Inteligencia Artificial para explorar datos del censo PAPE-AO*")
        
        st.divider()
        
        st.subheader("🔐 Iniciar Sesión")

                
        usuario = st.text_input("Usuario", placeholder="Ingresa tu usuario")
        contraseña = st.text_input("Contraseña", type="password", placeholder="Ingresa tu contraseña")
        
        col1, col2 = st.columns(2)
        with col1:
            login_btn = st.button("Iniciar Sesión", use_container_width=True, type="primary")
        #with col2:
            #demo_btn = st.button("Modo Demo", use_container_width=True)
        
        if login_btn:
            if usuario and contraseña:
                if verificar_credenciales(usuario, contraseña):
                    st.session_state.autenticado = True
                    st.session_state.usuario = usuario
                    st.session_state.fecha_inicio_sesion = datetime.now()
                    st.success("✅ Bienvenido!")
                    st.rerun()
                else:
                    st.error("❌ Usuario o contraseña incorrectos")
            else:
                st.warning("⚠️ Completa todos los campos")
        
        #f demo_btn:
            #st.session_state.autenticado = True
            #st.session_state.usuario = "demo"
            #st.session_state.fecha_inicio_sesion = datetime.now()
            #st.info("ℹ️ Modo Demo activado (sin límite de consultas)")
            #st.rerun()
        
        st.divider()
        st.caption("💡 Contacta al administrador si no tienes credenciales")

# ========================================
# VERIFICAR AUTENTICACIÓN
# ========================================

if "autenticado" not in st.session_state:
    st.session_state.autenticado = False

if not st.session_state.autenticado:
    mostrar_pantalla_login()
    st.stop()

# ========================================
# CARGAR DATOS Y AGENTE (una sola vez)
# ========================================
@st.cache_resource
def cargar_agente():
    integrator = DataIntegrator()
    df_completo = integrator.cargar_y_unir_datasets()
    api_key = st.secrets["DEEPSEEK_API_KEY"]
    return AgenteAnaliticoLLM(df_completo, api_key)

@st.cache_data
def obtener_estadisticas_basicas():
    """Obtiene estadísticas básicas del dataset"""
    agente = st.session_state.agente
    try:
        stats = {
            "total_registros": len(agente.df),
            "columnas": len(agente.df.columns),
            "fecha_actualizacion": datetime.now().strftime("%d/%m/%Y %H:%M")
        }
        return stats
    except:
        return None

if "agente" not in st.session_state:
    st.session_state.agente = cargar_agente()
    st.session_state.messages = []
    st.session_state.consultas_realizadas = 0

# ========================================
# INTERFAZ PRINCIPAL
# ========================================

# Header con info del usuario
col1, col2, col3 = st.columns([0.5, 0.3, 0.2])
with col1:
    st.markdown('<h1 class="main-header">Análisis AI PAPE-AO</h1>', unsafe_allow_html=True)
    st.markdown("*Inteligencia Artificial para explorar datos del censo PAPE-AO*")

with col2:
    st.metric("Consultas realizadas", st.session_state.consultas_realizadas)

with col3:
    col_user, col_logout = st.columns(2)
    with col_user:
        st.caption(f"👤 {st.session_state.usuario}")
    with col_logout:
        if st.button("🚪", help="Cerrar sesión"):
            st.session_state.autenticado = False
            st.session_state.usuario = None
            st.session_state.messages = []
            st.session_state.consultas_realizadas = 0
            st.rerun()

st.divider()

# ========================================
# TABS PRINCIPALES
# ========================================

tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📊 Dashboard", "🔍 Búsqueda", "📥 Exportar"])

# ========================================
# TAB 1: CHAT
# ========================================
with tab1:
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Información del usuario
        st.subheader("👤 Información de Usuario")
        st.write(f"**Usuario:** {st.session_state.usuario}")
        if st.session_state.usuario != "demo":
            limite_alcanzado, consultas_restantes = verificar_limite_consultas()
            limite_total = obtener_limite_consultas()
            
            st.write(f"**Consultas diarias:** {st.session_state.consultas_realizadas}/{limite_total}")
            
            progreso = st.session_state.consultas_realizadas / limite_total
            st.progress(progreso, text=f"{consultas_restantes} consultas restantes")
            
            if not limite_alcanzado:
                st.warning(f"⚠️ ¡Límite alcanzado!", icon="🔴")
        #else:
            #st.info("ℹ️ Modo Demo (sin límites)", icon="🎭")
        
        st.divider()
        
        # Información del dataset
        st.subheader("📈 Estado del Dataset")
        stats = obtener_estadisticas_basicas()
        if stats:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Registros", f"{stats['total_registros']:,}")
            with col2:
                st.metric("Columnas", stats['columnas'])
            st.caption(f"Actualizado: {stats['fecha_actualizacion']}")
        
        st.divider()
        
        # Ejemplos de preguntas
        st.subheader("💡 Ejemplos de Preguntas")
        example_prompts = [
            "¿Cuántas personas hay en total",
            "Cuántas mujeres de 60 a 64 años",
            "Cuántas personas con carencia a salud en Jalalpa el Grande?",
            "Tabla cruzada por sexo y edad de rezago educativo",
            "Personas elegibles Desde la Cuna en AGEB 11453"
        ]
        
        for i, example in enumerate(example_prompts, 1):
            if st.button(f"📌 {example}", key=f"example_{i}", use_container_width=True):
                st.session_state.prompt_sugerido = example
        
        st.divider()
        
        # Opciones adicionales
        st.subheader("🛠️ Opciones")
        temperatura = st.slider("Nivel de creatividad (temperatura)", 0.0, 1.0, 0.7, 0.1)
        limpiar_chat = st.button("🗑️ Limpiar chat", use_container_width=True)
        
        if limpiar_chat:
            st.session_state.messages = []
            st.session_state.consultas_realizadas = 0
            st.success("Chat limpiado")
            st.rerun()
        
        st.divider()
        st.markdown("""
            <div style='text-align: center; color: gray; font-size: 0.75rem; margin-top: 2rem;'>
                <p>💡 Inteligencia Artificial para la política social</p>
                <p><strong>Versión 2.1</strong></p>
                <p style='font-size: 0.7rem; margin-top: 1rem;'>Data & AI Inclusion Tech para Álcaldia Álavaro Obregón</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Mostrar historial
    st.subheader("💬 Conversación")
    st.divider()
    
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Generar respuesta del agente
    if "prompt_temp" in st.session_state:
        prompt = st.session_state.prompt_temp
        
        limite_alcanzado, consultas_restantes = verificar_limite_consultas()
        
        if not limite_alcanzado and st.session_state.usuario != "demo":
            st.error("❌ Has alcanzado tu límite de consultas diarias. Intenta mañana.")
        else:
            with st.chat_message("assistant"):
                with st.spinner("🤖 Analizando tu pregunta..."):
                    try:
                        respuesta = st.session_state.agente.procesar_consulta_mejorado(prompt)
                        st.markdown(respuesta)
                    except Exception as e:
                        st.error(f"❌ Error al procesar la consulta: {str(e)}")
                        respuesta = f"Error: {str(e)}"
            
            st.session_state.messages.append({"role": "assistant", "content": respuesta})
            
            # Mostrar sugerencias
            st.divider()
            st.subheader("💡 Preguntas sugeridas:")
            sugerencias = generar_sugerencias(prompt)
            cols = st.columns(len(sugerencias))
            for idx, sugerencia in enumerate(sugerencias):
                with cols[idx]:
                    if st.button(sugerencia, key=f"sugerencia_{idx}", use_container_width=True):
                        st.session_state.prompt_sugerido = sugerencia
                        st.rerun()
        
        del st.session_state.prompt_temp
    
    # Input del usuario
    st.divider()
    
    default_prompt = st.session_state.get("prompt_sugerido", "")
    if default_prompt:
        st.session_state.prompt_sugerido = ""
        st.session_state.messages.append({"role": "user", "content": default_prompt})
        st.session_state.consultas_realizadas += 1
        st.session_state.prompt_temp = default_prompt
        st.rerun()
    else:
        limite_alcanzado, consultas_restantes = verificar_limite_consultas()
        
        if limite_alcanzado or st.session_state.usuario == "demo":
            prompt = st.chat_input("Haz una pregunta sobre política social...")
            if prompt:
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.consultas_realizadas += 1
                st.session_state.prompt_temp = prompt
                st.rerun()
        else:
            st.error(f"❌ Límite diario alcanzado. {consultas_restantes} consultas restantes para mañana.")
            st.info("📞 Contacta al administrador si necesitas aumentar tu límite.")

# ========================================
# TAB 2: DASHBOARD
# ========================================
with tab2:
    st.subheader("📊 Dashboard de KPIs")
    
    try:
        agente = st.session_state.agente
        df = agente.df
        
        # KPIs principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total de Registros", f"{len(df):,}")
        
        with col2:
            st.metric("Columnas de Datos", len(df.columns))
        
        with col3:
            st.metric("Filas Completas", f"{df.dropna().shape[0]:,}")
        
        with col4:
            completitud = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("Completitud (%)", f"{completitud:.1f}%")
        
        st.divider()
        
        # Información del dataset
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Columnas del Dataset")
            for col in df.columns:
                dtype = str(df[col].dtype)
                st.caption(f"• **{col}** ({dtype})")
        
        with col2:
            st.subheader("📈 Estadísticas Básicas")
            st.write(df.describe().round(2))
    
    except Exception as e:
        st.error(f"Error cargando dashboard: {str(e)}")

# ========================================
# TAB 3: BÚSQUEDA
# ========================================
with tab3:
    st.subheader("🔍 Búsqueda y Filtros en Dataset")
    
    try:
        agente = st.session_state.agente
        df = agente.df
        
        # Buscador de texto
        search_text = st.text_input("🔎 Buscar en el dataset", placeholder="Ingresa texto para buscar...")
        
        if search_text:
            # Buscar en todas las columnas
            mask = df.astype(str).apply(lambda x: x.str.contains(search_text, case=False, na=False)).any(axis=1)
            resultados = df[mask]
            
            st.success(f"✅ {len(resultados)} registros encontrados")
            st.dataframe(resultados, use_container_width=True)
            
            # Descargar resultados
            csv = resultados.to_csv(index=False)
            st.download_button(
                label="📥 Descargar resultados (CSV)",
                data=csv,
                file_name=f"busqueda_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("💡 Ingresa un término de búsqueda para explorar el dataset")
        
        st.divider()
        
        # Filtros avanzados
        st.subheader("⚙️ Filtros Avanzados")
        
        col1, col2 = st.columns(2)
        
        with col1:
            columna_filtro = st.selectbox("Selecciona una columna", df.columns)
        
        with col2:
            if df[columna_filtro].dtype == 'object':
                valores_unicos = df[columna_filtro].dropna().unique()
                valor_filtro = st.multiselect("Valores", valores_unicos)
            else:
                min_val, max_val = float(df[columna_filtro].min()), float(df[columna_filtro].max())
                valor_filtro = st.slider("Rango", min_val, max_val, (min_val, max_val))
        
        # Aplicar filtro
        if columna_filtro:
            if df[columna_filtro].dtype == 'object' and valor_filtro:
                df_filtrado = df[df[columna_filtro].isin(valor_filtro)]
            elif df[columna_filtro].dtype != 'object':
                df_filtrado = df[(df[columna_filtro] >= valor_filtro[0]) & (df[columna_filtro] <= valor_filtro[1])]
            else:
                df_filtrado = df
            
            st.success(f"✅ {len(df_filtrado)} registros coinciden con el filtro")
            st.dataframe(df_filtrado, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error en búsqueda: {str(e)}")

# ========================================
# TAB 4: EXPORTAR
# ========================================
with tab4:
    st.subheader("📥 Exportar Conversación y Datos")
    
    if st.session_state.messages:
        col1, col2 = st.columns(2)
        
        # Exportar como JSON
        with col1:
            st.write("**📋 Conversación completa (JSON)**")
            json_data = json.dumps(st.session_state.messages, ensure_ascii=False, indent=2)
            st.download_button(
                label="📥 Descargar JSON",
                data=json_data,
                file_name=f"conversacion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # Exportar como PDF
        with col2:
            st.write("**📄 Conversación completa (PDF)**")
            if REPORTLAB_AVAILABLE:
                pdf_buffer = crear_pdf_reporte(st.session_state.messages, st.session_state.usuario)
                if pdf_buffer:
                    st.download_button(
                        label="📥 Descargar PDF",
                        data=pdf_buffer,
                        file_name=f"conversacion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
            else:
                st.warning("⚠️ ReportLab no instalado. Instala con: pip install reportlab")
        
        st.divider()
        
        # Exportar dataset filtrado
        st.subheader("📊 Exportar Dataset")
        agente = st.session_state.agente
        df = agente.df
        
        formato = st.selectbox("Selecciona formato", ["CSV", "Excel"])
        
        if formato == "CSV":
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Descargar Dataset (CSV)",
                data=csv_data,
                file_name=f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            buffer = BytesIO()
            df.to_excel(buffer, index=False, sheet_name="Datos")
            buffer.seek(0)
            st.download_button(
                label="📥 Descargar Dataset (Excel)",
                data=buffer,
                file_name=f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.info("💡 No hay conversación para exportar. ¡Comienza a hacer preguntas!")