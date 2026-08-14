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

# 3. MAPEO DEFINITIVO DESDE TEAMS.CSV
@st.cache_data
def cargar_diccionario_equipos_oficial(df_partidos):
    archivo_teams = 'teams.csv'
    
    if os.path.exists(archivo_teams):
        try:
            df_teams = pd.read_csv(archivo_teams)
            # Detectar automáticamente las columnas de ID y de Nombre
            col_id = [c for c in df_teams.columns if 'id' in c.lower() or 'uid' in c.lower()][0]
            col_name = [c for c in df_teams.columns if 'name' in c.lower() or 'display' in c.lower()][0]
            
            # Crear diccionario { ID: "Nombre Real" }
            return dict(zip(df_teams[col_id].astype(int), df_teams[col_name].astype(str)))
        except Exception as e:
            st.warning(f"Aviso técnico al procesar teams.csv: {e}")
            
    # Si no existe, usa IDs por defecto
    ids_unicos = set(df_partidos['homeTeamId'].unique()) | set(df_partidos['awayTeamId'].unique())
    return {int(id_eq): f"Club Deportivo (ID: {id_eq})" for id_eq in ids_unicos}

if not df_partidos.empty:
    diccionario_nombres = cargar_diccionario_equipos_oficial(df_partidos)
else:
    diccionario_nombres = {}

# 4. INTERFAZ INTERACTIVA CONTROLADA
if modelo is not None and not df_partidos.empty:
    
    # Obtener todos los IDs reales que tienen partidos registrados
    ids_con_partidos = set(df_partidos['homeTeamId'].unique()) | set(df_partidos['awayTeamId'].unique())
    
    # Construir la lista de opciones usando los nombres reales del archivo teams.csv
    opciones_menu = []
    for id_eq in ids_con_partidos:
        id_int = int(id_eq)
        nombre_verdadero = diccionario_nombres.get(id_int, f"Club Deportivo (ID: {id_int})")
        opciones_menu.append(f"{nombre_verdadero} [ID: {id_int}]")
        
    opciones_menu = sorted(opciones_menu)
    
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
        probabilidades = modelo.predict_proba(datos_entrada)
        
        prob_gana_local = probabilidades[0][1]
        
        st.subheader("📊 Diagnóstico del Modelo")
        if prob_gana_local > 0.55:
            st.success(f"🏆 Pronóstico: Victoria recomendada para el **{nombre_home}**")
        elif prob_gana_local < 0.45:
            st.warning(f"🛡️ Pronóstico: Doble oportunidad recomendada para **Empate o Victoria de {nombre_away}**")
        else:
            st.info("⚖️ Partido sumamente equilibrado, estadísticas muy parejas.")
            
        st.progress(float(prob_gana_local), text=f"Probabilidad de Victoria de {nombre_home}: {prob_gana_local:.2%}")
else:
    st.warning("⚠️ Esperando configuración de archivos en el repositorio de GitHub.")
