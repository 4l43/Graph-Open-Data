import pandas as pd
import io

# Ouvrir le fichier en mode texte avec gestion des erreurs
with open("Ecole.csv", "r", encoding="utf-8", errors="replace") as file:
    data = file.read()

# Lire le CSV à partir de la chaîne de caractères
df = pd.read_csv(io.StringIO(data), sep=";", encoding="utf-8")

# Vérifier si la colonne 'nb_etab_elem' existe avant de la manipuler
if 'nb_etab_elem' in df.columns:
    df['nb_etab_elem'] = df['nb_etab_elem'].fillna(0).astype(int)
else:
    raise KeyError("La colonne 'nb_etab_elem' est absente du fichier.")

# S'assurer que 'codgeo' et 'libgeo' sont bien des chaînes de caractères
df[['codgeo', 'libgeo']] = df[['codgeo', 'libgeo']].astype(str)

# Liste complète des codes géographiques et noms des communes
all_locations = df[['codgeo', 'libgeo']].drop_duplicates()

# Vérifier que la colonne 'secteur_ensgn' existe
if 'secteur_ensgn' in df.columns:
    # Création des colonnes pour public et privé
    df_public = df[df['secteur_ensgn'] == 'PUBLIC'].groupby(['codgeo', 'libgeo'])['nb_etab_elem'].sum().reset_index()
    df_public.rename(columns={'nb_etab_elem': 'nb_ecoles_publique'}, inplace=True)

    df_prive = df[df['secteur_ensgn'] == 'PRIVE'].groupby(['codgeo', 'libgeo'])['nb_etab_elem'].sum().reset_index()
    df_prive.rename(columns={'nb_etab_elem': 'nb_ecoles_prive'}, inplace=True)
else:
    raise KeyError("La colonne 'secteur_ensgn' est absente du fichier.")

# Fusionner avec la liste complète des communes pour inclure celles sans école
df_final = pd.merge(all_locations, df_public, on=['codgeo', 'libgeo'], how='left').fillna(0)
df_final = pd.merge(df_final, df_prive, on=['codgeo', 'libgeo'], how='left').fillna(0)

# Convertir les colonnes de compte en entier
df_final['nb_ecoles_publique'] = df_final['nb_ecoles_publique'].astype(int)
df_final['nb_ecoles_prive'] = df_final['nb_ecoles_prive'].astype(int)

# Ajouter une colonne pour le total
df_final['nb_ecoles_totale'] = df_final['nb_ecoles_publique'] + df_final['nb_ecoles_prive']

# Sauvegarder le résultat avec UTF-8
df_final.to_csv("Ecole_Aggregé.csv", index=False, sep=";", encoding="utf-8-sig")


# Afficher les premières lignes du fichier agrégé
print(df_final.head())
