import streamlit as st
import pandas as pd
import joblib
import os

# 1. Configuración de la interfaz gráfica
st.set_page_config(page_title="IA Predictora Avanzada", page_icon="⚽", layout="centered")
st.title("⚽ IA Predictora de Fútbol Inteligente")
st.write("Análisis estadístico basado en el historial completo del dataset de ESPN.")

# 2. CARGA SEGURA DEL MODELO (.pkl)
@st.cache_resource
def cargar_modelo():
    if os.path.exists('modelo_futbol_ia.pkl'):
        try:
            return joblib.load('modelo_futbol_ia.pkl')
        except Exception as e:
            st.error(f"Error técnico al abrir el archivo del modelo: {e}")
            return None
    else:
        st.error("⚠️ No se encontró 'modelo_futbol_ia.pkl' en el repositorio de GitHub.")
        return None

modelo = cargar_modelo()

# 3. CARGA ULTRA-RÁPIDA DEL DATASET LOCAL (.csv)
@st.cache_data
def cargar_datos_locales():
    if os.path.exists('datos_partidos_limpios.csv'):
        try:
            df = pd.read_csv('datos_partidos_limpios.csv')
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            return df
        except Exception as e:
            st.error(f"Error al leer el archivo de partidos: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

df_partidos = cargar_datos_locales()

# 4. INTERFAZ INTERACTIVA CON EQUIPOS REALES
if modelo is not None and not df_partidos.empty:
    # Extraer de forma única todos los IDs de equipos disponibles en el historial
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
        
    # Función para calcular la racha exacta basándose en el historial real
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

    # Ejecución automática de las rachas según la selección del usuario
    racha_home_calculada = calcular_racha_actual_equipo(df_partidos, id_home)
    racha_away_calculada = calcular_racha_actual_equipo(df_partidos, id_away)
    
    st.write(f"📈 **Racha detectada automáticamente (últimos 5 partidos):**")
    st.write(f"- {seleccion_home}: **{racha_home_calculada} pts**")
    st.write(f"- {seleccion_away}: **{racha_away_calculada} pts**")
    
    st.markdown("---")
    
    # 5. Botón ejecutor de la Predicción
    if st.button("🔮 Predecir Resultado con IA Real", use_container_width=True):
        datos_entrada = pd.DataFrame([[racha_home_calculada, racha_away_calculada]], columns=['home_racha_5', 'away_racha_5'])
        probabilidades = modelo.predict_proba(datos_entrada)
        
        prob_no_gana_local = probabilidades[0][0]  # Empate o Visitante
        prob_gana_local = probabilidades[0][1]     # Gana Local
        
        st.subheader("📊 Diagnóstico del Modelo")
        if prob_gana_local > 0.55:
            st.success(f"🏆 Pronóstico: Victoria recomendada para el **{seleccion_home}**")
        elif prob_no_gana_local > 0.55:
            st.warning(f"🛡️ Pronóstico: Doble oportunidad recomendada para **Empate o Victoria del {seleccion_away}**")
        else:
            st.info("⚖️ Partido sumamente equilibrado, estadísticas muy parejas.")
            
        st.progress(float(prob_gana_local), text=f"Probabilidad de Victoria Local: {prob_gana_local:.2%}")
else:
    st.warning("⚠️ Esperando archivos: Asegúrate de que 'datos_partidos_limpios.csv' y 'modelo_futbol_ia.pkl' estén correctamente subidos a tu repositorio de GitHub para activar los menús.")
