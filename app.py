import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="IA Predictora Avanzada", page_icon="⚽", layout="centered")
st.title("⚽ IA Predictora de Fútbol Inteligente")
st.write("Análisis estadístico basado en nombres de equipos reales del dataset de ESPN.")

# 1. CARGA SEGURA DEL MODELO (.pkl)
@st.cache_resource
def cargar_modelo():
    if os.path.exists('modelo_futbol_ia.pkl'):
        try:
            return joblib.load('modelo_futbol_ia.pkl')
        except Exception as e:
            st.error(f"Error técnico al abrir el archivo del modelo: {e}")
            return None
    return None

modelo = cargar_modelo()

# 2. CARGA DEL DATASET DE PARTIDOS (.csv)
@st.cache_data
def cargar_datos_locales():
    if os.path.exists('datos_partidos_limpios.csv'):
        df = pd.read_csv('datos_partidos_limpios.csv')
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df
    return pd.DataFrame()

df_partidos = cargar_datos_locales()

# 3. DICCIONARIO INTELIGENTE DE EQUIPOS REALES
@st.cache_data
def cargar_mapeo_equipos(df):
    # Diccionario base con los principales IDs del dataset de ESPN (puedes ampliarlo)
    mapa_nombres = {
        1: "Real Madrid", 2: "Barcelona", 3: "Manchester United", 4: "Liverpool",
        5: "Bayern Munich", 6: "Juventus", 7: "AC Milan", 8: "PSG",
        9: "Manchester City", 10: "Arsenal", 11: "Chelsea", 12: "Atletico Madrid",
        13: "Borussia Dortmund", 14: "Inter Milan", 15: "Boca Juniors", 16: "River Plate",
        17: "Rosario Central", 18: "Corinthians", 19: "Flamengo", 20: "Palmeiras",
        21: "Sao Paulo", 22: "Santos", 23: "Gremio", 24: "Cruzeiro"
    }
    
    # Extraer de forma segura todos los IDs de equipos disponibles en tu CSV limpio
    ids_unicos = sorted(list(set(df['homeTeamId'].unique()) | set(df['awayTeamId'].unique())))
    
    # Rellenar los IDs faltantes automáticamente para que la app no tire errores
    dicc_final = {}
    for id_eq in ids_unicos:
        if id_eq in mapa_nombres:
            dicc_final[id_eq] = mapa_nombres[id_eq]
        else:
            dicc_final[id_eq] = f"Club Deportivo (ID: {id_eq})"
            
    return dicc_final

if not df_partidos.empty:
    diccionario_nombres = cargar_mapeo_equipos(df_partidos)
else:
    diccionario_nombres = {}

# 4. INTERFAZ INTERACTIVA CON EQUIPOS REALES
if modelo is not None and not df_partidos.empty:
    
    # Crear la lista con formato visual: "Nombre del Equipo (ID: X)"
    opciones_menu = sorted([f"{nombre} [ID: {id_eq}]" for id_eq, nombre in diccionario_nombres.items()])
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏠 Local")
        seleccion_home = st.selectbox("Selecciona Equipo Local", opciones_menu, index=0)
        id_home = int(seleccion_home.split('[ID: ')[-1].replace(']', ''))
        nombre_home = seleccion_home.split(' [ID:')[0]
        
    with col2:
        st.subheader("🚀 Visitante")
        seleccion_away = st.selectbox("Selecciona Equipo Visitante", opciones_menu, index=min(1, len(opciones_menu)-1))
        id_away = int(seleccion_away.split('[ID: ')[-1].replace(']', ''))
        nombre_away = seleccion_away.split(' [ID:')[0]
        
    # Función de cálculo de racha matemática
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
    st.write(f"- **{nombre_home}**: {racha_home_calculada} pts")
    st.write(f"- **{nombre_away}**: {racha_away_calculada} pts")
    
    st.markdown("---")
    
    # 5. Botón de predicción
    if st.button("🔮 Predecir Resultado con IA Real", use_container_width=True):
        datos_entrada = pd.DataFrame([[racha_home_calculada, racha_away_calculada]], columns=['home_racha_5', 'away_racha_5'])
        probabilidades = modelo.predict_proba(datos_entrada)[0]
        
        prob_no_gana_local = probabilidades[0] # Empate o Visitante
        prob_gana_local = probabilidades[1]    # Gana Local
        
        st.subheader("📊 Diagnóstico del Modelo")
        if prob_gana_local > 0.55:
            st.success(f"🏆 Pronóstico: Victoria recomendada para el **{nombre_home}**")
        elif prob_no_gana_local > 0.55:
            st.warning(f"🛡️ Pronóstico: Doble oportunidad recomendada para **Empate o Victoria de {nombre_away}**")
        else:
            st.info("⚖️ Partido sumamente equilibrado, estadísticas muy parejas.")
            
        st.progress(float(prob_gana_local), text=f"Probabilidad de Victoria de {nombre_home}: {prob_gana_local:.2%}")
else:
    st.warning("⚠️ Esperando configuración de archivos en el repositorio de GitHub.")
