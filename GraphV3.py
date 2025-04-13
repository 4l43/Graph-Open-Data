import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community
import numpy as np
from matplotlib.lines import Line2D
from scipy.spatial.distance import pdist, squareform

# Charger les données (sélection de 200 lignes aléatoires)
df_full = pd.read_csv('ValeursYab.csv', sep=';')
df = df_full.sample(n=200, random_state=74)  # Sélection aléatoire de 200 lignes

# Liste des paramètres à considérer pour les arêtes
parametres = [
    'pop1-0',
    'nb_ecoles_publique1-0',
    'nb_ecoles_publique/pop1-0',
    'nb_ecoles_prive1-0',
    'nb/ecoles_prive/pop1-0',
    'nb_ecoles_totale1-0',
    'nb_ecoles_totale/pop1-0',
    'Part de la population éloignée de plus de 20 minutes d\'au moins un des services de santé de proximité 2021  1-0',
    'somme_tauxpourmille_crime 1-0',
    'Taux_Chomage 1-0',
    'Points',
    'Points 0-1'
]

# Vérifier et nettoyer les paramètres
valid_params = [p for p in parametres if p in df.columns]
print(f"Paramètres valides trouvés: {valid_params}")

# Préparer les données numériques (remplacer 'E' par 0 et convertir en float)
df_num = df.copy()
for param in valid_params:
    # Remplacer 'E' ou toute chaîne non numérique par 0
    df_num[param] = pd.to_numeric(df_num[param], errors='coerce').fillna(0)

# Calculer la matrice de distance de Gower
def gower_distance(X):
    """Calcule la distance de Gower entre toutes les paires d'observations"""
    n_samples, n_features = X.shape
    
    # Calculer les ranges pour normaliser
    ranges = np.zeros(n_features)
    for i in range(n_features):
        ranges[i] = np.max(X[:, i]) - np.min(X[:, i])
        if ranges[i] == 0:  # Éviter division par zéro
            ranges[i] = 1
    
    # Calculer les distances de Gower entre chaque paire
    gower_mat = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            # Somme des différences absolues normalisées
            diff_sum = 0
            valid_features = 0
            
            for k in range(n_features):
                # Si les deux valeurs sont valides
                if not (np.isnan(X[i, k]) or np.isnan(X[j, k])):
                    # Différence absolue normalisée
                    diff_sum += abs(X[i, k] - X[j, k]) / ranges[k]
                    valid_features += 1
            
            # Moyenne des différences si au moins une caractéristique valide
            if valid_features > 0:
                gower_mat[i, j] = diff_sum / valid_features
                gower_mat[j, i] = gower_mat[i, j]  # Symétrie
    
    return gower_mat

# Préparer la matrice des paramètres
param_matrix = df_num[valid_params].values

# Calculer la matrice de distance de Gower
distance_matrix = gower_distance(param_matrix)

# Convertir en matrice de similarité (1 - distance)
similarity_matrix = 1 - distance_matrix

# Définir un seuil de similarité moins strict pour créer plus de communautés
threshold = 0.1  # Seuil réduit pour obtenir davantage de connexions

# Créer une matrice d'adjacence à partir de la matrice de similarité
adjacency_matrix = similarity_matrix.copy()
# Mettre à zéro les similarités inférieures au seuil
adjacency_matrix[adjacency_matrix < threshold] = 0
# Enlever les boucles (connexions d'un nœud vers lui-même)
np.fill_diagonal(adjacency_matrix, 0)

# Créer le graphe à partir de la matrice d'adjacence
G = nx.from_numpy_array(adjacency_matrix)

# Renommer les nœuds avec les codegeo (pour la gestion interne)
mapping = {i: df['codegeo'].iloc[i] for i in range(len(df))}
G = nx.relabel_nodes(G, mapping)

# Ajouter les attributs aux nœuds
for node in G.nodes():
    idx = df.index[df['codegeo'] == node].tolist()[0]
    G.nodes[node]['libgeo'] = df.loc[idx, 'libgeo']
    G.nodes[node]['gagnant'] = df.loc[idx, 'Gagnant']

# Filtrer les arêtes avec un poids de 0
for u, v, d in list(G.edges(data=True)):
    if d['weight'] == 0:
        G.remove_edge(u, v)

# Si on a trop d'arêtes, ne garder que les plus fortes
max_edges = 1000 # A changer si on veut plus de connexions 
if G.number_of_edges() > max_edges:
    # Trier les arêtes par poids décroissant
    edges_with_weights = [(u, v, d['weight']) for u, v, d in G.edges(data=True)]
    edges_with_weights.sort(key=lambda x: x[2], reverse=True)
    
    # Ne garder que les max_edges arêtes les plus fortes
    edges_to_keep = edges_with_weights[:max_edges]
    
    # Recréer le graphe avec seulement ces arêtes
    G_new = nx.Graph()
    G_new.add_nodes_from(G.nodes(data=True))
    for u, v, w in edges_to_keep:
        G_new.add_edge(u, v, weight=w)
    G = G_new

# Détection des communautés (Louvain)
if G.number_of_edges() > 0:
    partition = community.best_partition(G)
else:
    print("Aucune arête créée avec ce seuil. Ajustez le seuil de similarité.")
    partition = {node: 0 for node in G.nodes()}  # Communauté par défaut

# Dictionnaire pour associer chaque parti gagnant à une couleur
partis_uniques = df['Gagnant'].unique()
couleurs_partis = {}
for i, parti in enumerate(partis_uniques):
    couleurs_partis[parti] = plt.cm.tab10(i % 10)  # Utilisation de la palette tab10

# Visualisation
plt.figure(figsize=(20, 15), dpi=300)

# Layout avec répulsion pour éviter la formation de clusters trop denses
pos = nx.spring_layout(G, k=0.5, iterations=100, seed=42)  # k élevé = plus de répulsion

# Coloration des nœuds selon le gagnant
node_colors = []
for node in G.nodes():
    idx = df.index[df['codegeo'] == node].tolist()[0]
    gagnant = df.loc[idx, 'Gagnant']
    node_colors.append(couleurs_partis[gagnant])

# Colorer les nœuds selon leur communauté (bordure)
node_community_colors = [plt.cm.viridis(partition[node]) for node in G.nodes()]

# Dessin des nœuds
nx.draw_networkx_nodes(G, pos, node_size=200, node_color=node_colors, alpha=0.8)

# Les arêtes sont dessinées avec une épaisseur proportionnelle au poids
if G.number_of_edges() > 0:
    edge_weights = [G[u][v]['weight'] * 3 for u, v in G.edges()]  # Multiplier par 3 pour mieux voir
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.4, edge_color='grey')

# Labels avec libgeo au lieu de codegeo
labels = {node: df.loc[df['codegeo'] == node, 'libgeo'].iloc[0] for node in G.nodes()}
# Ajuster la taille des labels selon la longueur du texte
font_sizes = {node: max(4, min(8, 12 - len(label)/3)) for node, label in labels.items()}
for node, label in labels.items():
    nx.draw_networkx_labels(G, pos, {node: label}, font_size=font_sizes[node])

# Légende pour les partis politiques
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                          label=parti, markersize=10) for parti, color in couleurs_partis.items()]
plt.legend(handles=legend_elements, title="Partis politiques", loc="upper right")

plt.title(f"Réseau de communes par similarité (Distance de Gower)\n"
          f"Arêtes créées si similarité ≥ {threshold} (max {max_edges} arêtes)\n"
          f"Couleurs des nœuds = Parti gagnant | Groupe = Communautés", fontsize=12)
plt.axis('off')

# Sauvegarder en PNG
plt.savefig('reseau_communes_gower_optimise.png', bbox_inches='tight', dpi=300)
plt.close()

# Afficher quelques statistiques sur le réseau
print(f"Nombre de nœuds: {G.number_of_nodes()}")
print(f"Nombre d'arêtes: {G.number_of_edges()}")
print(f"Nombre de communautés détectées: {len(set(partition.values()))}")

# Examiner le degré moyen et la distribution des degrés
if G.number_of_nodes() > 0:
    degrees = [d for n, d in G.degree()]
    if degrees:
        print(f"Degré moyen: {sum(degrees)/len(degrees):.2f}")
        print(f"Degré maximum: {max(degrees)}")
        print(f"Degré minimum: {min(degrees)}")

# Afficher les communautés
communautes = {}
for node, community_id in partition.items():
    if community_id not in communautes:
        communautes[community_id] = []
    communautes[community_id].append(node)

# print("\nListe des communautés détectées:")
# for comm_id, nodes in communautes.items():
#     communes = [df.loc[df['codegeo'] == node, 'libgeo'].iloc[0] for node in nodes]
#     print(f"Communauté {comm_id}: {', '.join(communes)}")

# Calcul et affichage du poids total de chaque communauté
print("\nListe des communautés détectées avec leur poids total :")
for comm_id, nodes in communautes.items():
    # Créer le sous-graphe correspondant à la communauté
    subG = G.subgraph(nodes)
    # Calculer le poids total pour les arêtes internes à la communauté
    poids_total = sum(d['weight'] for u, v, d in subG.edges(data=True))
    # Récupérer les libellés des communes pour affichage
    communes = [df.loc[df['codegeo'] == node, 'libgeo'].iloc[0] for node in nodes]
    print(f"Communauté {comm_id} : {', '.join(communes)}\n  Poids total des arêtes internes: {poids_total:.2f}\n")
