import streamlit as st
import joblib
import pandas as pd

# 1. Configuración de la página web
st.set_page_config(page_title="IA Predictora de Fútbol", page_icon="⚽", layout="centered")

st.title("⚽ Inteligencia Artificial Predictora de Fútbol")
st.write("Calcula la probabilidad de victoria basándote en el nivel y racha actual de los equipos.")

# 2. Cargar el cerebro de la IA de forma segura
@st.cache_resource
def cargar_modelo():
    try:
        return joblib.load('modelo_futbol_ia.pkl')
    except Exception as e:
        st.error(f"Error al cargar el archivo de la IA: {e}")
        return None

modelo = cargar_modelo()

if modelo is not None:
    st.info("💡 Tip: Para calcular la racha sumas: 3 pts por victoria, 1 pt por empate, 0 pts por derrota.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏠 Equipo Local")
        nombre_home = st.text_input("Nombre del Local", "Real Madrid")
        # El máximo de puntos en 5 partidos es 15 (3x5)
        racha_home = st.slider(f"Puntos últimos 5 partidos ({nombre_home})", 0, 15, 9)
        
    with col2:
        st.subheader("🚀 Equipo Visitante")
        nombre_away = st.text_input("Nombre del Visitante", "Barcelona")
        racha_away = st.slider(f"Puntos últimos 5 partidos ({nombre_away})", 0, 15, 6)
        
    st.markdown("---")
    
    # 3. Botón para ejecutar la predicción de la IA
    if st.button("🔮 Predecir Resultado con IA", use_container_width=True):
        # Crear la estructura de datos que el modelo espera recibir (mismos nombres de columnas que en el entrenamiento)
        datos_partido = pd.DataFrame([[racha_home, racha_away]], columns=['home_racha_5', 'away_racha_5'])
        
        # Obtener las probabilidades exactas del resultado
        probabilidades = modelo.predict_proba(datos_partido)[0]
        prob_no_gana_local = probabilidades[0] # Clase 0 (Empate o Gana Visitante)
        prob_gana_local = probabilidades[1]    # Clase 1 (Gana Local)
        
        st.subheader("📊 Resultado del Análisis del Modelo")
        
        if prob_gana_local > 0.55:
            st.success(f"🏆 Victoria recomendada para: **{nombre_home}**")
        elif prob_no_gana_local > 0.55:
            st.warning(f"🛡️ Doble oportunidad recomendada: **Empate o Victoria de {nombre_away}**")
        else:
            st.info("⚖️ Partido sumamente cerrado. Pronóstico reservado o de alto riesgo.")
            
        # Mostrar métricas visuales con barras de porcentaje
        st.write(f"📈 Probabilidad de victoria de {nombre_home} (Local): **{prob_gana_local:.2%}**")
        st.write(f"📉 Probabilidad de Empate o Victoria de {nombre_away}: **{prob_no_gana_local:.2%}**")
else:
    st.error("Por favor, asegúrate de colocar el archivo 'modelo_futbol_ia.pkl' en la misma carpeta que este script.")
