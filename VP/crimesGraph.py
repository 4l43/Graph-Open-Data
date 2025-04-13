import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random

def generer_graphe_candidats(fichier_donnees, nb_noeuds_max=1000, tolerance=5):
    # Charger le fichier fusionné
    df_merged = pd.read_csv(fichier_donnees, sep=None, engine='python')
    
    # Nettoyer la colonne "CodeInsee" et la renommer en "codegeo"
    df_merged.rename(columns={df_merged.columns[0]: "codegeo"}, inplace=True)
    df_merged["codegeo"] = df_merged["codegeo"].astype(str).str.zfill(5)
    
    # Définir des couleurs pour chaque candidat
    couleurs_candidats = {
        "LE PEN": "blue",
        "MACRON": "yellow",
        "FILLON": "red",
        "MÉLENCHON": "green",
        "LASSALLE": "purple",
        "DUPONT-AIGNAN": "orange",
        "HAMON": "pink",
        "ASSELINEAU": "brown"
    }
    
    # Réduire le nombre de nœuds pour la lisibilité
    df_sampled = df_merged.sample(min(nb_noeuds_max, len(df_merged)), random_state=42)
    
    # Créer un graphe
    G = nx.Graph()
    nodes_colors = {}
    taux_crimes = {}
    
    for _, row in df_sampled.iterrows():
        codegeo = row["codegeo"]
        gagnant = row["Gagnant"]
        couleur = couleurs_candidats.get(gagnant, "gray")  # Couleur grise si inconnu
        taux_crime = row["somme_tauxpourmille"]  # Supposons que la colonne s'appelle "TauxCrimes"
        
        G.add_node(codegeo, color=couleur)
        nodes_colors[codegeo] = couleur
        taux_crimes[codegeo] = taux_crime
    
    # Ajouter des arêtes entre communes ayant des taux similaires
    codes_geo = list(taux_crimes.keys())
    for i in range(len(codes_geo)):
        for j in range(i + 1, len(codes_geo)):
            if abs(taux_crimes[codes_geo[i]] - taux_crimes[codes_geo[j]]) <= tolerance:
                G.add_edge(codes_geo[i], codes_geo[j])
    
    # Dessiner le graphe
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, node_color=[nodes_colors[n] for n in G.nodes], with_labels=False, node_size=50, edge_color="gray", alpha=0.5)
    
    # Ajouter les annotations des taux de crimes
    taux_labels = {node: f"{taux_crimes[node]:.1f}" for node in G.nodes}
    # nx.draw_networkx_labels(G, pos, labels=taux_labels, font_size=8, font_color="black")
    plt.title("Graphe des communes coloré par candidat gagnant et connecté par taux de crimes")
    plt.savefig("CrimesGraph.png", dpi=300)
    print("Graph saved as CrimesGraph.png")

# Exemple d'utilisation
generer_graphe_candidats("Donnees_Fusionnees.csv")