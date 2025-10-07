import streamlit as st
import pandas as pd
import numpy as np

st.title("📊 Estimation avec stratification")

# --- Import du fichier ---
uploaded_file = st.file_uploader("Choisis un fichier CSV ou Excel", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("### Aperçu des données :")
    st.dataframe(df.head())

    # --- Sélection de la colonne de stratification ---
    strat_col = st.selectbox("Choisis la variable de stratification (ex: Type Avions IATA, Zone, etc.)", df.columns)

    # --- Sélection de la variable à analyser ---
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    selected_col = st.selectbox("Choisis la variable numérique à analyser :", numeric_cols)

    if strat_col and selected_col:
        st.markdown(f"### 🔹 Analyse de **{selected_col}** selon la stratification **{strat_col}**")

        # Supprimer les valeurs manquantes
        df = df.dropna(subset=[strat_col, selected_col])

        # --- Taille de chaque strate ---
        strata_sizes = df[strat_col].value_counts()
        Nh = strata_sizes
        N = len(df)

        # --- Moyenne et variance par strate ---
        strata_stats = df.groupby(strat_col)[selected_col].agg(['mean', 'var', 'count']).rename(columns={
            'mean': 'moyenne_h',
            'var': 'variance_h',
            'count': 'nh'
        })

        # --- Pondération par taille de strate ---
        strata_stats["poids_h"] = strata_stats["nh"] / N

        # --- Moyenne stratifiée ---
        stratified_mean = np.sum(strata_stats["poids_h"] * strata_stats["moyenne_h"])

        # --- Variance de l’estimateur stratifié ---
        stratified_variance = np.sum((strata_stats["poids_h"] ** 2) * (strata_stats["variance_h"] / strata_stats["nh"]))
        stratified_std = np.sqrt(stratified_variance)

        # --- Intervalle de confiance 95% ---
        z = 1.96
        lower_bound = stratified_mean - z * stratified_std
        upper_bound = stratified_mean + z * stratified_std

        # --- Résultats ---
        st.subheader("📈 Résultats globaux")
        st.write(f"**Moyenne stratifiée :** {stratified_mean:.3f}")
        st.write(f"**Écart-type de l’estimation :** {stratified_std:.3f}")
        st.write(f"**IC à 95% :** [{lower_bound:.3f} , {upper_bound:.3f}]")

        # --- Tableau récapitulatif ---
        st.subheader("📊 Statistiques par strate")
        st.dataframe(strata_stats)

        # --- Visualisation ---
        st.bar_chart(strata_stats["moyenne_h"])
