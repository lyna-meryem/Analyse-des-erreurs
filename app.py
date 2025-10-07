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
    st.info(f" Conversion automatique de {selected_delta_col} en kilogrammes (kg)")
    df[selected_delta_col] = df[selected_delta_col] * 1000
# ---------------------------

# S’assurer que la colonne choisie est bien numérique
df[selected_delta_col] = pd.to_numeric(df[selected_delta_col], errors="coerce")

# ---------------------------
# 3. Filtres (dans la sidebar à droite)
# ---------------------------
st.sidebar.header("Filtres")

# ----- CityPair -----
city_options = sorted(df["[LIDO] Citypair"].dropna().unique().tolist())

if "selected_cities" not in st.session_state:
    st.session_state.selected_cities = []

search_city = st.sidebar.text_input("Rechercher un CityPair")

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

st.write(f"Nombre de vols filtrés : **{len(df_filtered)}**")
# ==========================
# ANALYSE DE LA DISTRIBUTION DE L'ÉCHANTILLON
# ==========================
st.header("2. Analyse de la Distribution de l'Échantillon")

if len(df_filtered) > 0:
    # --- Vérification que la colonne est numérique ---
    if np.issubdtype(df_filtered[selected_delta_col].dtype, np.number):
        data = df_filtered[selected_delta_col].dropna()

        if len(data) > 1:
            # --- Graphique : histogramme + densité ---
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(data, bins=30, color="skyblue", edgecolor="black", alpha=0.7, density=True)

            # Courbe de densité empirique
            from scipy.stats import gaussian_kde
            density = gaussian_kde(data)
            x_vals = np.linspace(data.min(), data.max(), 200)
            ax.plot(x_vals, density(x_vals), color="red", linewidth=2, label="Densité empirique")

            # Courbe normale théorique
            from scipy.stats import norm
            p = norm.pdf(x_vals, data.mean(), data.std())
            ax.plot(x_vals, p, 'green', linestyle="--", label="Loi normale théorique")

            ax.set_title(f"Distribution de {selected_delta_col}")
            ax.set_xlabel(selected_delta_col)
            ax.set_ylabel("Densité")
            ax.legend()
            st.pyplot(fig)

            # --- Statistiques descriptives ---
            st.subheader("Statistiques descriptives")
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
            st.warning("⚠️ Trop peu de données pour effectuer l’analyse.")
    else:
        st.warning(f"⚠️ La colonne '{selected_delta_col}' n’est pas numérique.")
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
    
# ---------------------------
# 3. Choix de la stratification
# ---------------------------
strat_col1 = st.selectbox("Stratifier par", ["Type Avions IATA", "Area"])
strat_col2 = st.selectbox("Optionnelle: deuxième stratification", ["Aucune", "Type Avions IATA", "Area"])

# ---------------------------
# 4. Calculs par strat
# ---------------------------
def strat_analysis(df, value_col, strat_cols):
    results = []
    N_total = len(df)
    
    for name, group in df.groupby(strat_cols):
        N_h = len(group)  # population de la strate
        n_h = N_h        # si l'échantillon = population de la strate
        mean_h = group[value_col].mean()
        var_h = group[value_col].var(ddof=1)
        
        results.append({
            "Strate": name,
            "Nh": N_h,
            "nh": n_h,
            "Moyenne": mean_h,
            "Variance": var_h
        })
    
    res_df = pd.DataFrame(results)
    
    # Moyenne globale stratifiée
    res_df["Pondération"] = res_df["Nh"] / res_df["Nh"].sum()
    mean_global = np.sum(res_df["Moyenne"] * res_df["Pondération"])
    
    # Variance globale stratifiée
    res_df["Var_Pondérée"] = (res_df["Pondération"]**2) * (res_df["Variance"]) / res_df["nh"] 
    var_global = res_df["Var_Pondérée"].sum()
    
    return res_df, mean_global, var_global

# Stratification principale
strat_cols = [strat_col1]
if strat_col2 != "Aucune" and strat_col2 != strat_col1:
    strat_cols.append(strat_col2)

# Δ
delta_res, delta_mean_global, delta_var_global = strat_analysis(df, selected_delta_col, strat_cols)

# LIDO (colonne précédente)
col_index = df.columns.get_loc(selected_delta_col)
lido_col = df.columns[col_index - 1]
df[lido_col] = pd.to_numeric(df[lido_col], errors="coerce")
lido_res, lido_mean_global, lido_var_global = strat_analysis(df, lido_col, strat_cols)

# Erreur relative par strate


# Erreur relative par strate (IC 95%)
delta_res["Erreur_rel (%)"] = (np.sqrt(delta_res["Var_Pondérée"]) * 1.96 / delta_res["Moyenne"]) * 100

# Erreur relative globale
global_erreur_rel = (np.sqrt(delta_var_global) * 1.96 / delta_mean_global) * 100


# ---------------------------
# 5. Affichage
# ---------------------------
st.subheader("Moyenne et variance par strate pour Delta")
st.write(delta_res)
st.subheader("Moyenne et variance par strate pour LIDO")
st.write(lido_res)

st.subheader("Moyenne et variance globales")
st.write(f"Moyenne Delta global = {delta_mean_global:.2f}, Var(Δ) global = {delta_var_global:.2f}")
st.write(f"Moyenne LIDO global = {lido_mean_global:.2f}, Var(LIDO) global = {lido_var_global:.2f}")

st.subheader("Erreur relative Δ / LIDO (%)")
st.write(delta_res[["Strate", "Erreur_rel (%)"]])
st.write(f"Erreur relative globale = {global_erreur_rel:.2f}%")
