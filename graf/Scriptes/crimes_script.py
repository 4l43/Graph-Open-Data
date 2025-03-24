import pandas as pd

# Charger le fichier
file_path = 'crimesUTF.csv'  # Remplacez par le chemin correct
data = pd.read_csv(file_path, encoding='utf-8', sep=';')

# Nettoyer les données
data = data.replace(r'\\N', pd.NA, regex=True)  # Remplacer \N par NaN
data['tauxpourmille'] = pd.to_numeric(data['tauxpourmille'], errors='coerce')  # Convertir en numérique

# Calculer la somme des taux pour mille par code
result = data.groupby('CODGEO_2024', as_index=False)['tauxpourmille'].sum()
result = result.rename(columns={'tauxpourmille': 'somme_tauxpourmille'})

# Sauvegarder le résultat dans un fichier CSV
output_path = 'somme_tauxpourmille_par_code.csv'  # Remplacez par le chemin désiré
result.to_csv(output_path, index=False, encoding='utf-8')

print(f"Le fichier résultat a été sauvegardé à : {output_path}")
