import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random



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
    "Type Avions IATA",git pull origin main --rebase

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

    st.subheader("📌 Analyse des Outliers")
    st.write("Vols outliers (méthode 3σ) :", outliers_sigma)
    st.write("Vols outliers (méthode IQR) :", outliers_iqr)



# Trouver la colonne LIDO qui précède la colonne Delta choisie
col_index = df_filtered.columns.get_loc(selected_delta_col)
mean_observed = df_filtered[selected_delta_col].mean()

if col_index > 0:
    lido_col = df_filtered.columns[col_index - 1]  # juste avant
    df_filtered[lido_col] = pd.to_numeric(df_filtered[lido_col], errors="coerce")

    # Moyenne LIDO
    mean_lido = df_filtered[lido_col].mean()

    # Pourcentages relatifs
    mean_pct = (mean_observed / mean_lido) * 100
    ci_low_pct = (ci_low / mean_lido) * 100
    ci_high_pct = (ci_high / mean_lido) * 100

    st.subheader("📊 Intervalle de confiance relatif (%)")
    st.write(f"Moyenne {lido_col} : **{mean_lido:.2f}**")
    st.write(f"Gain moyen relatif (Delta vs LIDO) : **{mean_pct:.2f}%**")
    st.write(f"IC95% relatif : **[{ci_low_pct:.2f}%, {ci_high_pct:.2f}%]**")
else:
    st.error("⚠️ Impossible de trouver la colonne LIDO correspondante.")
    
    
col_index = df_filtered.columns.get_loc(selected_delta_col)
lido_col = df_filtered.columns[col_index - 1]
df_filtered[lido_col] = df_filtered[lido_col].astype(str).str.replace('[\$,]', '', regex=True)
df_filtered[lido_col] = pd.to_numeric(df_filtered[lido_col], errors='coerce')


st.write("Colonne LIDO récupérée :", lido_col)
st.write("Premières lignes :", df_filtered[[lido_col, selected_delta_col]].head())

# ---------------------------
# 7. Méthode par strates (types d’avions, citypairs)
# ---------------------------
st.header("4. Méthode stratifiée (Types d’avions / Citypairs)")

# Choix de la strate
strata_col = st.selectbox(
    "Choisir la variable de stratification",
    ["Type Avions IATA", "[LIDO] Citypair"]
)

# Estimation stratifiée
def stratified_estimation(df, strata_col, value_col, N_dict, z=1.96):
    results = []
    for h, group in df.groupby(strata_col):
        nh = random.randint(1, len(group)) # échantillon
        Nh = N_dict.get(h, nh) # population réelle (à fournir si connue)
        xh_bar = group[value_col].mean()      # moyenne observée
        sh = group[value_col].std(ddof=1)     # écart-type

        Th = Nh * xh_bar                      # total estimé
        var_Th = (Nh**2 * sh**2 / nh) * ((Nh - nh) / (Nh - 1)) if nh > 1 else 0

        results.append({
            "Strate": h,
            "Nh (total vols)": Nh,
            "nh (échantillon)": nh,
            "Moyenne échantillon": xh_bar,
            "Total estimé (Th)": Th,
            "Variance(Th)": var_Th
        })

    res_df = pd.DataFrame(results)
    T_hat = res_df["Total estimé (Th)"].sum()
    Var_T = res_df["Variance(Th)"].sum()
    SE_T = np.sqrt(Var_T)
    ME = z * SE_T
    IC_low, IC_high = T_hat - ME, T_hat + ME

    return res_df, T_hat, Var_T, (IC_low, IC_high)

# ⚠️ À remplacer par les vrais Nh si tu les connais (nombre total de vols réels par strate)
N_dict = {h: len(g) for h, g in df_filtered.groupby(strata_col)}

if len(df_filtered) > 0:
    res_df, T_hat, Var_T, IC = stratified_estimation(df_filtered, strata_col, selected_delta_col, N_dict)

    st.subheader("📊 Résultats par strate")
    st.dataframe(res_df)

    st.write(f"**Total estimé** : {T_hat:.2f}")
    st.write(f"**Variance totale** : {Var_T:.2f}")
    st.write(f"**IC95% du total** : [{IC[0]:.2f}, {IC[1]:.2f}]")
else:
    st.warning("⚠️ Aucune donnée disponible pour la stratification.")

