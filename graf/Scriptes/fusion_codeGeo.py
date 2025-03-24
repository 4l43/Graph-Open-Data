# import pandas as pd

# def fusionner_donnees(fichier_ecole, fichier_medecin, fichier_crimes, fichier_sortie):
#     # Charger les fichiers
#     df_ecole = pd.read_csv(fichier_ecole, sep=';')
#     df_medecin = pd.read_csv(fichier_medecin, sep=None, engine='python')
#     df_crimes = pd.read_csv(fichier_crimes, sep=None, engine='python')
    
#     # Nettoyer les colonnes "codegeo"
#     df_ecole.rename(columns={"codgeo": "codegeo"}, inplace=True)
#     df_ecole["codegeo"] = df_ecole["codegeo"].astype(str).str.strip()
#     df_medecin.rename(columns={df_medecin.columns[0]: "codegeo"}, inplace=True)
#     df_medecin["codegeo"] = df_medecin["codegeo"].astype(str).str.zfill(5)
#     df_crimes.rename(columns={df_crimes.columns[0]: "codegeo"}, inplace=True)
#     df_crimes["codegeo"] = df_crimes["codegeo"].astype(str).str.zfill(5)
    
#     # Fusionner les fichiers
#     df_final = df_ecole.merge(df_medecin, on="codegeo", how="outer").merge(df_crimes, on="codegeo", how="outer")
    
#     # Supprimer les colonnes redondantes
#     if "Libellé" in df_final.columns and "libgeo" in df_final.columns:
#         df_final.drop(columns=["Libellé"], inplace=True)
    
#     # Sauvegarder le fichier fusionné
#     df_final.to_csv(fichier_sortie, index=False,sep=";", encoding="utf-8-sig")
#     print(f"Fichier fusionné enregistré sous : {fichier_sortie}")

# # Exemple d'utilisation
# fusionner_donnees("Ecole_Aggregé.csv", "Medecin.csv", "Crimes_Taux.csv", "Donnees_Fusionnees.csv")


import pandas as pd

def fusionner_donnees(fichier_ecole, fichier_medecin, fichier_crimes, fichier_presidentiel, fichier_sortie):
    # Charger les fichiers
    df_ecole = pd.read_csv(fichier_ecole, sep=';')
    df_medecin = pd.read_csv(fichier_medecin, sep=None, engine='python')
    df_crimes = pd.read_csv(fichier_crimes, sep=None, engine='python')
    df_presidentiel = pd.read_csv(fichier_presidentiel, sep=None, engine='python')
    
    # Nettoyer les colonnes "codegeo"
    df_ecole.rename(columns={"codgeo": "codegeo"}, inplace=True)
    df_ecole["codegeo"] = df_ecole["codegeo"].astype(str).str.strip()
    df_medecin.rename(columns={df_medecin.columns[0]: "codegeo"}, inplace=True)
    df_medecin["codegeo"] = df_medecin["codegeo"].astype(str).str.zfill(5)
    df_crimes.rename(columns={df_crimes.columns[0]: "codegeo"}, inplace=True)
    df_crimes["codegeo"] = df_crimes["codegeo"].astype(str).str.zfill(5)
    df_presidentiel.rename(columns={df_presidentiel.columns[0]: "codegeo"}, inplace=True)
    df_presidentiel["codegeo"] = df_presidentiel["codegeo"].astype(str).str.zfill(5)
    
    # Fusionner les fichiers
    df_final = df_ecole.merge(df_medecin, on="codegeo", how="outer").merge(df_crimes, on="codegeo", how="outer").merge(df_presidentiel, on="codegeo", how="left")
    
    # Supprimer les colonnes redondantes
    if "Libellé" in df_final.columns and "libgeo" in df_final.columns:
        df_final.drop(columns=["Libellé"], inplace=True)
    
    # Sauvegarder le fichier fusionné
    df_final.to_csv(fichier_sortie, index=False,sep=";", encoding="utf-8-sig")
    print(f"Fichier fusionné enregistré sous : {fichier_sortie}")

# Exemple d'utilisation
fusionner_donnees("Ecole_Aggregé.csv", "Medecin.csv", "Crimes_Taux.csv", "Presidentiel.csv", "Donnees_Fusionnees.csv")
