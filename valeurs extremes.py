import streamlit as st
import pandas as pd

st.title("Détection des valeurs extrêmes par type d’avion")

# 🔹 Charger le fichier
uploaded_file = st.file_uploader("Choisis un fichier CSV ou Excel", type=["csv", "xlsx"])

if uploaded_file:
    # Lecture du fichier
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("### Aperçu des données :")
    st.dataframe(df.head())

    # Vérifier la présence de la colonne 'Type Avions IATA'
    if "Type Avions IATA" not in df.columns:
        st.error("La colonne 'Type Avions IATA' est absente du fichier. Merci de vérifier le nom exact.")
    else:
        # Nettoyage : remplacement des virgules par des points et conversion en numérique
        for col in df.columns:
            
                
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Sélection de la colonne numérique à analyser
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        selected_col = st.selectbox("Choisir une colonne numérique :", numeric_cols)

        if selected_col:
            # Fonction pour détecter les outliers dans chaque groupe (type avion)
            def detect_outliers_iqr(group):
                Q1 = group.quantile(0.25)
                Q3 = group.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                return (group < lower) | (group > upper)

            # 🔹 Application par type d’avion
            df["is_outlier"] = df.groupby("Type Avions IATA")[selected_col].transform(detect_outliers_iqr)

            # 🔹 Filtrer les valeurs extrêmes
            outliers_df = df[df["is_outlier"] == True]

            # 🔹 Colonnes à afficher : les 5 premières + Type Avions IATA + selected_col
            base_cols = df.columns[:5].tolist()
            extra_cols = ["Commercial Flight Number", "Type Avions IATA", selected_col]

            # Retirer les doublons et garder uniquement les colonnes existantes
            cols_to_display = [col for col in dict.fromkeys(base_cols + extra_cols) if col in df.columns]

            # 🔹 Affichage du tableau final
            st.subheader(f"Valeurs extrêmes pour '{selected_col}' par type d’avion")
            st.write(f"Nombre total de valeurs extrêmes : {len(outliers_df)}")
            st.dataframe(outliers_df[cols_to_display])

            # 🔹 Téléchargement CSV
            csv = outliers_df[cols_to_display].to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Télécharger les valeurs extrêmes",
                data=csv,
                file_name="valeurs_aberrantes_par_type_avion.csv",
                mime="text/csv"
            )
