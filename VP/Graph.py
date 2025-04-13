import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random

def generer_graphe_candidats(fichier_presidentiel, nb_noeuds_max=1000):
    # Charger le fichier
    df_presidentiel = pd.read_csv(fichier_presidentiel, sep=None, engine='python')
    
    # Nettoyer la colonne "CodeInsee" et la renommer en "codegeo"
    df_presidentiel.rename(columns={df_presidentiel.columns[0]: "codegeo"}, inplace=True)
    df_presidentiel["codegeo"] = df_presidentiel["codegeo"].astype(str).str.zfill(5)
    
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
    df_sampled = df_presidentiel.sample(min(nb_noeuds_max, len(df_presidentiel)), random_state=42)
    
    # Créer un graphe
    G = nx.Graph()
    nodes_colors = []
    for _, row in df_sampled.iterrows():
        codegeo = row["codegeo"]
        gagnant = row["Gagnant"]
        couleur = couleurs_candidats.get(gagnant, "gray")  # Couleur grise si inconnu
        
        G.add_node(codegeo, color=couleur)
        nodes_colors.append(couleur)
    
    # Ajouter des arêtes aléatoires
    random_edges = random.sample(list(G.nodes), min(500, len(G.nodes)))
    for i in range(len(random_edges) - 1):
        G.add_edge(random_edges[i], random_edges[i + 1])
    
    # Dessiner le graphe
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, node_color=nodes_colors, with_labels=False, node_size=50, edge_color="gray", alpha=0.5)
    plt.title("Graphe des régions coloré par candidat gagnant (échantillon réduit)")
    # plt.show()
    plt.savefig("graph_actors_movies.png", dpi=300)
    
# Exemple d'utilisation
generer_graphe_candidats("Donnees_Fusionnees.csv")
