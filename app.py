import streamlit as st
import pandas as pd
import joblib
import os
import zipfile
import urllib.request
import base64

st.set_page_config(page_title="IA Predictora Avanzada", page_icon="⚽", layout="centered")
st.title("⚽ IA Predictora de Fútbol Inteligente")
st.write("Conexión automática a Kaggle mediante descarga directa ultraligera.")

# 1. CARGA SEGURA DEL MODELO
@st.cache_resource
def cargar_modelo_seguro():
    try:
        return joblib.load('modelo_futbol_ia.pkl')
    except Exception as e:
        st.error(f"Error al abrir el archivo del modelo .pkl: {e}")
        return None

modelo = cargar_modelo_seguro()

# 2. DESCARGA DIRECTA CORREGIDA CON HEADERS DE AUTENTICACIÓN
@st.cache_data(show_spinner="🔄 Descargando y actualizando estadísticas desde Kaggle de forma directa...")
def descargar_datos_directo():
    if "KAGGLE_USERNAME" in st.secrets and "KAGGLE_KEY" in st.secrets:
        usuario = st.secrets["KAGGLE_USERNAME"]
        llave = st.secrets["KAGGLE_KEY"]
    else:
        st.warning("⚠️ Falta configurar las credenciales KAGGLE_USERNAME y KAGGLE_KEY en los Secrets de Streamlit.")
        return pd.DataFrame()

    ruta_zip = "dataset.zip"
    ruta_extraccion = "base_data"
    os.makedirs(ruta_extraccion, exist_ok=True)
    
    url = "https://kaggle.com"
    
    try:
        # Crear la cabecera de autenticación básica requerida por la API de Kaggle
        credenciales = f"{usuario}:{llave}".encode("utf-8")
        auth_base64 = base64.b64encode(credenciales).decode("utf-8")
        
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Basic {auth_base64}")
        
        # Realizar la descarga del archivo binario comprimido de forma correcta
        with urllib.request.urlopen(req) as respuesta, open(ruta_zip, "wb") as archivo_salida:
            archivo_salida.write(respuesta.read())
        
        # Descomprimir los archivos CSV
        with zipfile.ZipFile(ruta_zip, 'r') as zip_ref:
            zip_ref.extractall(ruta_extraccion)
            
        if os.path.exists(ruta_zip):
            os.remove(ruta_zip)
            
    except Exception as e:
        st.error(f"Error al descargar directamente de Kaggle: {e}")
        return pd.DataFrame()
    
    # Buscar todos los archivos CSV extraídos
    todos_los_archivos = []
    for raiz, dirs, archivos in os.walk(ruta_extraccion):
        for archivo in archivos:
            if archivo.endswith('.csv') and 'base_data' in raiz:
                todos_los_archivos.append(os.path.join(raiz, archivo))
                
    if not todos_los_archivos:
        st.warning("No se encontraron tablas estructuradas de partidos en la carpeta.")
        return pd.DataFrame()
        
    try:
        lista_df = [pd.read_csv(f, low_memory=False, engine='python') for f in todos_los_archivos]
        df = pd.concat(lista_df, ignore_index=True)
        df = df.dropna(subset=['date', 'homeTeamId', 'awayTeamId', 'homeTeamWinner'])
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error al estructurar las tablas de fútbol: {e}")
        return pd.DataFrame()

df_partidos = descargar_datos_directo()

# 3. INTERFAZ GRÁFICA CONTROLADA
if modelo is not None and not df_partidos.empty:
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
        probabilidades = modelo.predict_proba(datos_entrada)
        
        prob_no_gana_local = probabilidades[0][0]
        prob_gana_local = probabilidades[0][1]
        
        st.subheader("📊 Diagnóstico del Modelo")
        if prob_gana_local > 0.55:
            st.success(f"🏆 Pronóstico: Victoria recomendada para el **{seleccion_home}**")
        elif prob_no_gana_local > 0.55:
            st.warning(f"🛡️ Pronóstico: Doble oportunidad recomendada para **Empate o Victoria del {seleccion_away}**")
        else:
            st.info("⚖️ Partido sumamente equilibrado, estadísticas muy parejas.")
            
        st.progress(float(prob_gana_local), text=f"Probabilidad de Victoria Local: {prob_gana_local:.2%}")
else:
    st.info("💡 Iniciando la descarga y procesamiento seguro de datos...")
