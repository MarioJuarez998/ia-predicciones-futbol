import streamlit as st
import pandas as pd
import joblib
import os
import glob

# 1. Configuración inicial de la interfaz web
st.set_page_config(page_title="IA Predictora Avanzada", page_icon="⚽", layout="centered")
st.title("⚽ IA Predictora de Fútbol Inteligente")
st.write("Conexión automática a Kaggle. Selecciona equipos reales para calcular probabilidades.")

# 2. Descarga automática del dataset de Kaggle usando Secrets tradicionales
@st.cache_data(show_spinner="🔄 Conectando con Kaggle y actualizando estadísticas del día...")
def actualizar_base_datos():
    # Inyectar credenciales oficiales que lee la librería de Kaggle
    if "KAGGLE_USERNAME" in st.secrets and "KAGGLE_KEY" in st.secrets:
        os.environ['KAGGLE_USERNAME'] = st.secrets["KAGGLE_USERNAME"]
        os.environ['KAGGLE_KEY'] = st.secrets["KAGGLE_KEY"]
    
    os.makedirs('base_data', exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        # Descarga directa del dataset
        api.dataset_download_files('excel4soccer/espn-soccer-data', path='./base_data', unzip=True)
    except Exception as e:
        st.error(f"Error de conexión con Kaggle: {e}")
        return pd.DataFrame()
    
    # Consolidar y limpiar partidos
    archivos_csv = glob.glob(os.path.join('./base_data', '**/*.csv'), recursive=True)
    if not archivos_csv:
        return pd.DataFrame()
        
    lista_df = [pd.read_csv(f, low_memory=False) for f in archivos_csv]
    df = pd.concat(lista_df, ignore_index=True)
    df = df.dropna(subset=['date', 'homeTeamId', 'awayTeamId', 'homeTeamWinner'])
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    return df

# Ejecutar descarga
df_partidos = actualizar_base_datos()

# 3. Cargar el modelo predictivo (Tu Cerebro de IA)
@st.cache_resource
def cargar_modelo():
    try:
        return joblib.load('modelo_futbol_ia.pkl')
    except Exception as e:
        st.error(f"Error al cargar el archivo .pkl del modelo: {e}")
        return None

modelo = cargar_modelo()

# 4. Lógica de cálculo en vivo si los datos y el modelo están listos
if not df_partidos.empty and modelo is not None:
    
    # Extraer ID únicos de equipos disponibles
    equipos_disponibles = sorted(list(set(df_partidos['homeTeamId'].unique()) | set(df_partidos['awayTeamId'].unique())))
    opciones_equipos = {f"Equipo ID: {eq}": eq for eq in equipos_disponibles}
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏠 Local")
        seleccion_home = st.selectbox("Selecciona Equipo Local", list(opciones_equipos.keys()), index=0)
        id_home = opciones_equipos[seleccion_home]
        
    with col2:
        st.subheader("🚀 Visitante")
        seleccion_away = st.selectbox("Selecciona Equipo Visitante", list(opciones_equipos.keys()), index=min(1, len(opciones_equipos)-1))
        id_away = opciones_equipos[seleccion_away]
        
    def calcular_racha_actual_equipo(df, team_id, n_partidos=5):
        partidos_equipo = df[(df['homeTeamId'] == team_id) | (df['awayTeamId'] == team_id)]
        ultimos_partidos = partidos_equipo.tail(n_partidos)
        
        puntos = 0
        for _, row in ultimos_partidos.iterrows():
            if row['homeTeamId'] == team_id:
                if str(row['homeTeamWinner']).lower() == 'true': puntos += 3
            else:
                if str(row['homeTeamWinner']).lower() == 'false': puntos += 3
        return puntos

    racha_home_calculada = calcular_racha_actual_equipo(df_partidos, id_home)
    racha_away_calculada = calcular_racha_actual_equipo(df_partidos, id_away)
    
    st.write(f"📈 **Racha detectada automáticamente (últimos 5 partidos):**")
    st.write(f"- {seleccion_home}: **{racha_home_calculada} pts**")
    st.write(f"- {seleccion_away}: **{racha_away_calculada} pts**")
    
    st.markdown("---")
    
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
