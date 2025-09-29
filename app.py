import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 1. Charger les données
# ---------------------------
df = pd.read_csv("vols.csv", parse_dates=["[FK] Flight date"])

# ---------------------------
# 2. Sélection de la colonne Delta
# ---------------------------
delta_columns = [col for col in df.columns if "Delta" in col]
selected_delta_col = st.selectbox("Choisir la colonne Delta à analyser", delta_columns)

# S’assurer que la colonne est numérique
df[selected_delta_col] = pd.to_numeric(df[selected_delta_col], errors="coerce")

# ---------------------------
# 3. Filtres Streamlit
# ---------------------------

# ----- CityPair -----
# ----- CityPair -----
city_options = sorted(df["[LIDO] Citypair"].dropna().unique().tolist())

if "selected_cities" not in st.session_state:
    st.session_state.selected_cities = []

# Champ recherche
search_city = st.text_input("🔎 Rechercher un CityPair")

# Filtrer les options en fonction de la recherche
if search_city:
    filtered_city_options = [c for c in city_options if search_city.lower() in c.lower()]
else:
    filtered_city_options = city_options

def select_all_cities():
    st.session_state.selected_cities = filtered_city_options

st.button("Sélectionner tous les CityPairs affichés", on_click=select_all_cities)

selected_cities = st.multiselect(
    "CityPair",
    options=filtered_city_options,
    default=st.session_state.selected_cities
)


# ----- Type Avion -----
type_options = df["Type Avions IATA"].dropna().unique().tolist()
if "selected_types" not in st.session_state:
    st.session_state.selected_types = []

def select_all_types():
    st.session_state.selected_types = type_options

st.button("Sélectionner tous les Types Avions", on_click=select_all_types)

selected_types = st.multiselect(
    "Type Avions IATA",
    options=type_options,
    default=st.session_state.selected_types
)

# ----- Dates -----
date_range = st.date_input(
    "Sélectionner la période",
    [df["[FK] Flight date"].min(), df["[FK] Flight date"].max()]
)
start_date = pd.to_datetime(date_range[0])
end_date = pd.to_datetime(date_range[1])

# ----- Area -----
area_options = df["Area"].dropna().unique().tolist()
if "selected_area" not in st.session_state:
    st.session_state.selected_area = []

def select_all_area():
    st.session_state.selected_area = area_options

st.button("Sélectionner tous les secteurs", on_click=select_all_area)

selected_area = st.multiselect(
    "Area",
    options=area_options,
    default=st.session_state.selected_area
)

# ---------------------------
# 4. Filtrage des données
# ---------------------------
df_filtered = df[
    (df["[LIDO] Citypair"].isin(selected_cities)) &
    (df["Area"].isin(selected_area)) &
    (df["Type Avions IATA"].isin(selected_types)) &
    (df["[FK] Flight date"].between(start_date, end_date))
]

st.write(f"Nombre de vols filtrés : {len(df_filtered)}")

# ---------------------------
# 5. Bootstrap et IC95%
# ---------------------------
df_filtered = df_filtered.dropna(subset=[selected_delta_col])

if len(df_filtered) > 0:
    NBOOT = 5000
    boot_means = [
        df_filtered[selected_delta_col].sample(frac=1, replace=True).mean()
        for _ in range(NBOOT)
    ]
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    mean_observed = df_filtered[selected_delta_col].mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(boot_means, bins=30, color="skyblue", edgecolor="black")
    ax.axvline(ci_low, color="red", linestyle="--", label=f"IC 2.5% = {ci_low:.2f}")
    ax.axvline(ci_high, color="red", linestyle="--", label=f"IC 97.5% = {ci_high:.2f}")
    ax.axvline(mean_observed, color="green", linestyle="-", label=f"Moyenne = {mean_observed:.2f}")
    ax.set_title(f"Distribution bootstrap de {selected_delta_col}")
    ax.set_xlabel("Gain moyen (Delta)")
    ax.set_ylabel("Fréquence")
    ax.legend()
    st.pyplot(fig)
else:
    st.warning("⚠️ Aucun vol valide trouvé pour la colonne sélectionnée")

# ---------------------------
# 6. Analyse des outliers
# ---------------------------
if len(df_filtered) > 0:
    mean_delta = df_filtered[selected_delta_col].mean()
    std_delta = df_filtered[selected_delta_col].std()

    # Méthode 3 sigma
    outliers_sigma = df_filtered[np.abs(df_filtered[selected_delta_col] - mean_delta) > 3 * std_delta]

    # Méthode IQR
    Q1 = df_filtered[selected_delta_col].quantile(0.25)
    Q3 = df_filtered[selected_delta_col].quantile(0.75)
    IQR = Q3 - Q1
    outliers_iqr = df_filtered[
        (df_filtered[selected_delta_col] < Q1 - 1.5 * IQR) |
        (df_filtered[selected_delta_col] > Q3 + 1.5 * IQR)
    ]

    st.write("Vols outliers (méthode 3σ) :", outliers_sigma)
    st.write("Vols outliers (méthode IQR) :", outliers_iqr)


