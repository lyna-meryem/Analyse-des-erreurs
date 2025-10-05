import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

st.title("Méthodes Statistiques d'Analyse des Données")

st.header("1. Intervalle de Confiance (IC)")
st.write("La notion d'intervalle de confiance renvoie au degré de précision d’une moyenne ou d’un pourcentage. "
         "Elle s’appuie sur un échantillon et vise à estimer la fiabilité que l’on peut accorder aux valeurs observées "
         "par rapport aux valeurs réelles de la population totale.")

st.write("Formule générale :")
st.latex(r"IC = \bar{x} \pm z_{\alpha/2} \cdot \frac{s}{\sqrt{n}}")

st.write("Avec :")
st.latex(r"\overline{x} = \text{moyenne de l’échantillon}")
st.latex(r"s = \text{écart-type}")
st.latex(r"n = \text{taille de l’échantillon}")
st.latex(r"z_{\alpha/2} \approx 1.96 \text{ pour un IC à 95\%}")

# ---------------------------
# 1. Charger les données
# ---------------------------
df = pd.read_csv("vols.csv", parse_dates=["[FK] Flight date"])

# ---------------------------
# 2. Sélection de la colonne Delta
# ---------------------------
delta_columns = [col for col in df.columns if "Delta" in col]
selected_delta_col = st.selectbox("Choisir la colonne Delta à analyser", delta_columns)

df[selected_delta_col] = (
    df[selected_delta_col]
    .astype(str)
    .str.replace(",", ".", regex=False)
)
df[selected_delta_col] = pd.to_numeric(df[selected_delta_col], errors="coerce")

# Conversion en kg si nécessaire
if "en T" in selected_delta_col:
    st.info(f"⚖️ Conversion automatique de {selected_delta_col} en kilogrammes (kg)")
    df[selected_delta_col] = df[selected_delta_col] * 1000

# ---------------------------
# 3. Filtres dans la sidebar
# ---------------------------
st.sidebar.header("📌 Filtres")

# ----- CityPair -----
city_options = sorted(df["[LIDO] Citypair"].dropna().unique().tolist())
if "selected_cities" not in st.session_state:
    st.session_state.selected_cities = []

search_city = st.sidebar.text_input("🔎 Rechercher un CityPair")
filtered_city_options = [c for c in city_options if search_city.lower() in c.lower()] if search_city else city_options

def select_all_cities():
    st.session_state.selected_cities = filtered_city_options

st.sidebar.button("Sélectionner tous les CityPairs affichés", on_click=select_all_cities)

default_selected = [c for c in st.session_state.selected_cities if c in filtered_city_options]
selected_cities = st.sidebar.multiselect("CityPair", options=filtered_city_options, default=default_selected)

# ----- Type Avion -----
type_options = df["Type Avions IATA"].dropna().unique().tolist()
if "selected_types" not in st.session_state:
    st.session_state.selected_types = []

def select_all_types():
    st.session_state.selected_types = type_options

st.sidebar.button("Sélectionner tous les Types Avions", on_click=select_all_types)
selected_types = st.sidebar.multiselect("Type Avions IATA", options=type_options, default=st.session_state.selected_types)

# ----- Période -----
min_date = df["[FK] Flight date"].min()
max_date = df["[FK] Flight date"].max()
date_range = st.sidebar.slider("Sélectionner la période", min_value=min_date.to_pydatetime(),
                               max_value=max_date.to_pydatetime(),
                               value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
                               format="DD/MM/YYYY")
start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

# ----- Area -----
area_options = df["Area"].dropna().unique().tolist()
if "selected_area" not in st.session_state:
    st.session_state.selected_area = []

def select_all_area():
    st.session_state.selected_area = area_options

st.sidebar.button("Sélectionner tous les secteurs", on_click=select_all_area)
selected_area = st.sidebar.multiselect("Area", options=area_options, default=st.session_state.selected_area)

# ---------------------------
# 4. Filtrage des données
# ---------------------------
df_filtered = df[
    (df["[LIDO] Citypair"].isin(selected_cities)) &
    (df["Area"].isin(selected_area)) &
    (df["Type Avions IATA"].isin(selected_types)) &
    (df["[FK] Flight date"].between(start_date, end_date))
]

st.write(f"📊 Nombre de vols filtrés : **{len(df_filtered)}**")

df_filtered = df_filtered.dropna(subset=[selected_delta_col])

# ---------------------------
# 5. Distributions (comparatives + individuelles)
# ---------------------------
if len(df_filtered) > 0:
    st.subheader("📊 Visualisation des distributions")

    numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) > 0:
        # 🔸 Comparaison de plusieurs distributions
        selected_multi = st.multiselect("Comparer plusieurs variables :", numeric_cols, default=numeric_cols[:2])
        if selected_multi:
            fig, ax = plt.subplots(figsize=(10, 5))
            for col in selected_multi:
                ax.hist(df_filtered[col].dropna(), bins=30, alpha=0.5, edgecolor="black", label=f"{col} (μ={df_filtered[col].mean():.2f})")
            ax.set_title("Comparaison des distributions sélectionnées")
            ax.set_xlabel("Valeur")
            ax.set_ylabel("Fréquence")
            ax.legend()
            st.pyplot(fig)
            st.write(df_filtered[selected_multi].describe().T)

        # 🔸 Distribution individuelle
        selected_dist_col = st.selectbox("Visualiser une variable individuelle :", numeric_cols)
        fig, ax = plt.subplots(figsize=(8, 4))
        data = df_filtered[selected_dist_col].dropna()
        ax.hist(data, bins=30, color="skyblue", edgecolor="black", alpha=0.7)
        ax.axvline(data.mean(), color="red", linestyle="--", label=f"Moyenne = {data.mean():.2f}")
        ax.axvline(data.median(), color="green", linestyle=":", label=f"Médiane = {data.median():.2f}")
        ax.set_title(f"Distribution de {selected_dist_col}")
        ax.legend()
        st.pyplot(fig)
        st.write("📊 **Résumé statistique :**")
        st.write(data.describe().to_frame().T)
    else:
        st.warning("⚠️ Aucune colonne numérique disponible.")
else:
    st.warning("⚠️ Aucun vol valide trouvé pour la colonne sélectionnée.")
