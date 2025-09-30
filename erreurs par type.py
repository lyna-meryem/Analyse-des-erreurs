import pandas as pd
import numpy as np

def stratified_estimation(df, strata_col, value_col, N_dict, z=1.96):
    """
    Calcule l'estimateur stratifié (total, variance, IC95%)
    
    df : DataFrame filtré avec un échantillon
    strata_col : nom de la colonne des strates (ex: 'Type Avions IATA')
    value_col : colonne numérique à analyser (ex: 'Delta Fuel')
    N_dict : dictionnaire {stratum: N_h} donnant la taille réelle de chaque strate
    z : quantile normal (1.96 pour 95%)
    """

    results = []

    for h, group in df.groupby(strata_col):
        nh = len(group)                       # taille de l'échantillon
        Nh = N_dict.get(h, nh)                # taille réelle de la strate
        xh_bar = group[value_col].mean()      # moyenne observée
        sh = group[value_col].std(ddof=1)     # écart-type observé

        # Estimateur du total
        Th = Nh * xh_bar

        # Variance (approx sans remise)
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

    # Somme des totaux estimés
    T_hat = res_df["Total estimé (Th)"].sum()

    # Variance totale
    Var_T = res_df["Variance(Th)"].sum()

    # Erreur standard
    SE_T = np.sqrt(Var_T)

    # Marge d'erreur
    ME = z * SE_T

    IC_low, IC_high = T_hat - ME, T_hat + ME

    return res_df, T_hat, Var_T, (IC_low, IC_high)
