import streamlit as st
import pandas as pd
import joblib
import os
import glob

# 1. Configuración inicial de la interfaz web
st.set_page_config(page_title="IA Predictora Avanzada", page_icon="⚽", layout="centered")
st.title("⚽ IA Predictora de Fútbol Inteligente")
st.write("Conexión automática a Kaggle. Selecciona equipos reales para calcular probabilidades.")

# 2. Descarga automática del dataset de Kaggle usando el Token de Secrets
@st.cache_data(show_spinner="🔄 Conectando con Kaggle y actualizando estadísticas del día...")
def actualizar_base_datos():
    # Inyectar el token de los Secrets de Streamlit en las variables del sistema
    if "KAGGL_API_TOKEN" in st.secrets:
        os.environ['KAGGL_API_TOKEN'] = st.secrets["KAGGL_API_TOKEN"]
    
    # Crear directorio si no existe y descargar
    os.makedirs('base_data', exist_ok=True)
    try:
        # Comando python interno para descargar sin usar la terminal de comandos de consola
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files('excel4soccer/espn-soccer-data', path='./base_data', unzip=True)
    except Exception as e:
        st.warning(f"Usando datos locales cacheables. Detalle de conexión opcional: {e}")
    
    # Consolidar y limpiar los partidos en un solo DataFrame ordenado
    archivos_csv = glob.glob(os.path.join('./base_data', '**/*.csv'), recursive=True)
    if not archivos_csv:
        return pd.DataFrame()
        
    lista_df = [pd.read_csv(f, low_memory=False) for f in archivos_csv]
    df = pd.concat(lista_df, ignore_index=True)
    df = df.dropna(subset=['date', 'homeTeamId', 'awayTeamId', 'homeTeamWinner'])
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    return df

# Ejecutar descarga y procesamiento inicial
df_partidos = actualizar_base_datos()

# 3. Cargar el modelo predictivo (Tu Cerebro de IA)
@st.cache_resource
def cargar_modelo():
    return joblib.load('modelo_futbol_ia.pkl')

modelo = cargar_modelo()

# 4. Lógica de cálculo en vivo si los datos y el modelo están listos
if not df_partidos.empty and modelo is not None:
    
    # En este dataset los ID son números. Para mostrar nombres bonitos, crearemos un mapeo simplificado
    # Extramos todos los ID únicos de equipos disponibles
    equipos_disponibles = sorted(list(set(df_partidos['homeTeamId'].unique()) | set(df_partidos['awayTeamId'].unique())))
    
    # Creamos etiquetas amigables para el usuario: "Equipo ID: 123"
    opciones_equipos = {f"Equipo ID: {eq}": eq for eq in equipos_disponibles}
    
    # Interfaz de usuario con menús desplegables en lugar de cuadros de texto
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏠 Local")
        seleccion_home = st.selectbox("Selecciona Equipo Local", list(opciones_equipos.keys()), index=0)
        id_home = opciones_equipos[seleccion_home]
        
    with col2:
        st.subheader("🚀 Visitante")
        # El index=1 es para que no aparezca seleccionado el mismo equipo por defecto
        seleccion_away = st.selectbox("Selecciona Equipo Visitante", list(opciones_equipos.keys()), index=min(1, len(opciones_equipos)-1))
        id_away = opciones_equipos[seleccion_away]
        
    # --- FUNCIÓN MATEMÁTICA DE RACHAS EN TIEMPO REAL ---
    def calcular_racha_actual_equipo(df, team_id, n_partidos=5):
        # Filtrar todos los partidos donde participó este equipo específico
        partidos_equipo = df[(df['homeTeamId'] == team_id) | (df['awayTeamId'] == team_id)]
        # Tomar los últimos N partidos jugados históricamente en el dataset
        ultimos_partidos = partidos_equipo.tail(n_partidos)
        
        puntos = 0
        for _, row in ultimos_partidos.iterrows():
            if row['homeTeamId'] == team_id: # Jugó como local
                if str(row['homeTeamWinner']).lower() == 'true': puntos += 3
            else: # Jugó como visitante
                if str(row['homeTeamWinner']).lower() == 'false': puntos += 3
        return puntos

    # Calcular automáticamente los puntos de las rachas basados en el ID seleccionado
    racha_home_calculada = calcular_racha_actual_equipo(df_partidos, id_home)
    racha_away_calculada = calcular_racha_actual_equipo(df_partidos, id_away)
    
    # Mostrar al usuario qué racha detectó la IA de forma interna
    st.write(f"📈 **Racha detectada automáticamente (últimos 5 partidos):**")
    st.write(f"- {seleccion_home}: **{racha_home_calculada} pts**")
    st.write(f"- {seleccion_away}: **{racha_away_calculada} pts**")
    
    st.markdown("---")
    
    # 5. Ejecución del botón predictivo
    if st.button("🔮 Predecir Resultado con IA Real", use_container_width=True):
        datos_entrada = pd.DataFrame([[racha_home_calculada, racha_away_calculada]], columns=['home_racha_5', 'away_racha_5'])
        probabilidades = modelo.predict_proba(datos_entrada)[0]
        
        prob_no_gana_local = probabilidades[0]
        prob_gana_local = probabilidades[1]
        
        st.subheader("📊 Diagnóstico del Modelo")
        if prob_gana_local > 0.55:
            st.success(f"🏆 Pronóstico: Victoria recomendada para el **{seleccion_home}**")
        elif prob_no_gana_local > 0.55:
            st.warning(f"🛡️ Pronóstico: Doble oportunidad recomendada para **Empate o Victoria del {seleccion_away}**")
        else:
            st.info("⚖️ Partido sumamente equilibrado estadísticas muy parejas.")
            
        st.progress(float(prob_gana_local), text=f"Probabilidad de Victoria Local: {prob_gana_local:.2%}")
else:
    st.error("Error al procesar la base de datos o cargar el archivo del modelo '.pkl'. Verifica tus archivos de configuración.")
