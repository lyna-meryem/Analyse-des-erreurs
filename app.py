import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


st.title("Analyse des erreurs")

st.header("1. Intervalle de Confiance (IC)")
st.write("La notion d'intervalle de confiance renvoie au degré de précision d’une moyenne ou d’un pourcentage. "
         "Elle s’appuie sur un échantillon et vise à estimer la fiabilité que l’on peut accorder aux valeurs observées "
         "par rapport aux valeurs réelles de la population totale.")


# st.write("Formule générale :")
# st.latex(r"IC = \bar{x} \pm z_{\alpha/2} \cdot \frac{s}{\sqrt{n}}")
# st.write("Avec :")
# st.latex(r"\overline{x} = \text{moyenne de l’échantillon}")
# st.latex(r"s = \text{écart-type}")
# st.latex(r"n = \text{taille de l’échantillon}")
# st.latex(r"z_{\alpha/2} \approx 1.96 \text{ pour un IC à 95\%}")

# ---------------------------
# 1. Charger les données
# ---------------------------
df = pd.read_csv("vols.csv", parse_dates=["[FK] Flight date"])

# ---------------------------
# 2. Sélection de la colonne Delta

# ---------------------------
# 2. Sélection de la colonne Delta
# ---------------------------
delta_columns = [col for col in df.columns if "Delta" in col]
selected_delta_col = st.selectbox("Choisir la colonne Delta à analyser", delta_columns)

df[selected_delta_col] = (
    df[selected_delta_col]
    .astype(str)                # transformer en texte pour le nettoyage
    .str.replace(",", ".", regex=False)  # remplacer virgule par point
)
# S’assurer que la colonne choisie est bien numérique
df[selected_delta_col] = pd.to_numeric(df[selected_delta_col], errors="coerce")

# ---------------------------
# Conversion en kg si nécessaire
# ---------------------------
if "en T" in selected_delta_col:   # Si le nom de la colonne contient "T"
    st.info(f"⚖️ Conversion automatique de {selected_delta_col} en kilogrammes (kg)")
    df[selected_delta_col] = df[selected_delta_col] * 1000

# ---------------------------

# S’assurer que la colonne choisie est bien numérique
df[selected_delta_col] = pd.to_numeric(df[selected_delta_col], errors="coerce")

# ---------------------------
# 3. Filtres (dans la sidebar à droite)
# ---------------------------
st.sidebar.header("📌 Filtres")

# ----- CityPair -----
city_options = sorted(df["[LIDO] Citypair"].dropna().unique().tolist())


if "selected_cities" not in st.session_state:
    st.session_state.selected_cities = []

search_city = st.sidebar.text_input("🔎 Rechercher un CityPair")

if search_city:
    filtered_city_options = [c for c in city_options if search_city.lower() in c.lower()]
else:
    filtered_city_options = city_options

def select_all_cities():
    st.session_state.selected_cities = filtered_city_options

st.sidebar.button("Sélectionner tous les CityPairs affichés", on_click=select_all_cities)

# Filtrer les valeurs par défaut pour éviter l'erreur
default_selected = [c for c in st.session_state.selected_cities if c in filtered_city_options]

selected_cities = st.sidebar.multiselect(
    "CityPair",
    options=filtered_city_options,
    default=default_selected
)


# ----- Type Avion -----
type_options = df["Type Avions IATA"].dropna().unique().tolist()
if "selected_types" not in st.session_state:
    st.session_state.selected_types = []

def select_all_types():
    st.session_state.selected_types = type_options

st.sidebar.button("Sélectionner tous les Types Avions", on_click=select_all_types)

selected_types = st.sidebar.multiselect(
    "Type Avions IATA",
    options=type_options,
    default=st.session_state.selected_types
)

# ----- Dates avec un slider -----
min_date = df["[FK] Flight date"].min()
max_date = df["[FK] Flight date"].max()

date_range = st.sidebar.slider(
    "Sélectionner la période",
    min_value=min_date.to_pydatetime(),
    max_value=max_date.to_pydatetime(),
    value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
    format="DD/MM/YYYY"
)

start_date = pd.to_datetime(date_range[0])
end_date = pd.to_datetime(date_range[1])

# ----- Area -----
area_options = df["Area"].dropna().unique().tolist()
if "selected_area" not in st.session_state:
    st.session_state.selected_area = []

def select_all_area():
    st.session_state.selected_area = area_options

st.sidebar.button("Sélectionner tous les secteurs", on_click=select_all_area)

selected_area = st.sidebar.multiselect(
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

st.write(f"📊 Nombre de vols filtrés : **{len(df_filtered)}**")



# ==========================
# 📈 ANALYSE DE LA DISTRIBUTION DE L'ÉCHANTILLON
# ==========================
st.header("2. Analyse de la Distribution de l'Échantillon")

if len(df_filtered) > 0:
    # --- Sélection de la colonne à analyser ---
    numeric_cols = selected_delta_col.select_dtypes(include=[np.number]).columns.tolist()
    selected_dist_col = st.selectbox("Choisir une variable numérique à analyser", numeric_cols, index=0)

    data = df_filtered[selected_dist_col].dropna()

    # --- Graphique : histogramme + densité ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(data, bins=30, color="skyblue", edgecolor="black", alpha=0.7, density=True)
    
    # Courbe de densité
    from scipy.stats import gaussian_kde
    density = gaussian_kde(data)
    x_vals = np.linspace(data.min(), data.max(), 200)
    ax.plot(x_vals, density(x_vals), color="red", linewidth=2, label="Densité empirique")

    # Courbe normale théorique
    from scipy.stats import norm
    p = norm.pdf(x_vals, data.mean(), data.std())
    ax.plot(x_vals, p, 'green', linestyle="--", label="Loi normale théorique")

    ax.set_title(f"Distribution de {selected_dist_col}")
    ax.set_xlabel(selected_dist_col)
    ax.set_ylabel("Densité")
    ax.legend()
    st.pyplot(fig)

    # --- Statistiques descriptives ---
    st.subheader("📊 Statistiques descriptives")
    st.write(data.describe().to_frame().T)

    # --- Asymétrie et aplatissement ---
    from scipy.stats import skew, kurtosis
    skewness = skew(data)
    kurt = kurtosis(data)
    st.write(f"**Asymétrie (Skewness)** : {skewness:.3f}")
    st.write(f"**Aplatissement (Kurtosis)** : {kurt:.3f}")

    # --- Test de normalité (Shapiro-Wilk) ---
    from scipy.stats import shapiro
    stat, p_value = shapiro(data)
    st.subheader("🧪 Test de normalité (Shapiro-Wilk)")
    st.write(f"Statistique de test : {stat:.4f}")
    st.write(f"p-value : {p_value:.4f}")

    if p_value > 0.05:
        st.success("✅ L'échantillon suit une distribution normale (H₀ non rejetée).")
    else:
        st.warning("⚠️ L'échantillon ne suit pas une distribution normale (H₀ rejetée).")

else:
    st.warning("⚠️ Aucun échantillon valide trouvé pour effectuer l’analyse.")


















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

    # --- Plot distribution bootstrap ---
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





    # ---------------------------
    # 6. Analyse des outliers
    # ---------------------------
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

    st.subheader("📌 Analyse des Outliers")
    st.write("Vols outliers (méthode 3σ) :", outliers_sigma)
    st.write("Vols outliers (méthode IQR) :", outliers_iqr)
    

    # ---------------------------
    # IC relatif par rapport à LIDO
    # ---------------------------
    col_index = df_filtered.columns.get_loc(selected_delta_col)
    if col_index > 0:
        lido_col = df_filtered.columns[col_index - 1]
        df_filtered[lido_col] = pd.to_numeric(df_filtered[lido_col], errors="coerce")

        mean_lido = df_filtered[lido_col].mean()
        if mean_lido and not np.isnan(mean_lido) and mean_lido != 0:
            mean_pct = (mean_observed / mean_lido) * 100
            ci_low_pct = (ci_low / mean_lido) * 100
            ci_high_pct = (ci_high / mean_lido) * 100

            st.subheader("📊 Intervalle de confiance relatif (%)")
            st.write(f"Moyenne {lido_col} : **{mean_lido:.2f}**")
            st.write(f"Gain moyen relatif (Delta vs LIDO) : **{mean_pct:.2f}%**")
            st.write(f"IC95% relatif : **[{ci_low_pct:.2f}%, {ci_high_pct:.2f}%]**")
        else:
            st.warning("⚠️ Impossible de calculer le pourcentage relatif (moyenne LIDO invalide).")
    else:
        st.error("⚠️ Impossible de trouver la colonne LIDO correspondante.")
else:
    st.warning("⚠️ Aucun vol valide trouvé pour la colonne sélectionnée")