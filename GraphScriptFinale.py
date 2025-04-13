import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain
from matplotlib.lines import Line2D
from matplotlib import cm
import matplotlib.patches as mpatches
from tqdm import tqdm
from collections import defaultdict

# Configuration initiale
FICHIER = 'donnéefinalecorrigénormal.csv'

# Chargement du fichier
df = pd.read_csv(FICHIER, sep=';', encoding='latin-1', low_memory=False)

# Liste des paramètres
parametres = [
    'pop1-0',
    'nb_ecoles_publique/pop1-0',
    'nb/ecoles_prive/pop1-0',
    'nb_ecoles_totale/pop1-0',
    'Part de la population éloignée de plus de 20 minutes d\'au moins un des services de santé de proximité 2021  1-0',
    'somme_tauxpourmille_crime 1-0',
    'Taux_Chomage 1-0',
]

# Poids associés
poids_metriques = {
    'pop1-0': 0.1,
    'somme_tauxpourmille_crime 1-0': -0.3,
    'Taux_Chomage 1-0': -0.5,
    'nb_ecoles_totale/pop1-0': 0.1,
    'nb_ecoles_publique/pop1-0': 0.3,
    'nb/ecoles_prive/pop1-0': 0.3,
    'Part de la population éloignée de plus de 20 minutes d\'au moins un des services de santé de proximité 2021  1-0': 0.15,
}

# Nettoyage des données
for col in parametres:
    if col in df.columns:
        df[col] = (df[col].astype(str)
                   .str.replace(',', '.')
                   .str.replace(' ', '')
                   .replace(['FAUX', 'NA', 'NaN', 'None', 'null'], '0'))
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Assurez-vous que la colonne 'Gagnant' existe bien
if 'Gagnant' not in df.columns:
    raise KeyError("La colonne 'Gagnant' est absente du fichier.")

# Nettoyage de la colonne gagnant
df['Gagnant'] = df['Gagnant'].str.strip().str.upper()

# Filtrage
parametres_existants = [col for col in parametres if col in df.columns]
df_clean = df[['libgeo', 'Gagnant'] + parametres_existants].dropna()

# Échantillonnage
max_communes = 300
df_filtered = df_clean.sample(n=min(max_communes, len(df_clean)), random_state=42)

# Matrice de similarité pondérée
def calculate_weighted_similarity(data, columns, weights):
    n = len(data)
    similarity = np.zeros((n, n))
    weight_sum = sum(weights.values())
    norm_weights = {k: v / weight_sum for k, v in weights.items()}
    for i in tqdm(range(n)):
        for j in range(i+1, n):
            score = 0
            for col in columns:
                if col in norm_weights:
                    diff = abs(data[col].iloc[i] - data[col].iloc[j])
                    col_range = data[col].max() - data[col].min()
                    if col_range > 0:
                        normalized_diff = diff / col_range
                        score += norm_weights[col] * (1 - normalized_diff)
            similarity[i, j] = similarity[j, i] = score
    return similarity

similarity_matrix = calculate_weighted_similarity(df_filtered, parametres_existants, poids_metriques)

# Seuil de similarité - augmenté pour avoir un graphe moins dense
threshold = np.percentile(similarity_matrix[similarity_matrix > 0], 80)

# Création du graphe
G = nx.Graph()
communes = df_filtered['libgeo'].tolist()

# Ajouter les nœuds et les arêtes
for i, commune in enumerate(communes):
    data_row = df_filtered.iloc[i]
    G.add_node(commune, gagnant=data_row['Gagnant'], **{p: data_row[p] for p in parametres_existants})
    for j in range(i+1, len(communes)):
        if similarity_matrix[i, j] >= threshold:
            G.add_edge(commune, communes[j], weight=similarity_matrix[i, j])

# Retirer les nœuds isolés
isolated_nodes = list(nx.isolates(G))
G.remove_nodes_from(isolated_nodes)

# Détection des communautés
partition = community_louvain.best_partition(G, weight='weight')
communities = set(partition.values())

# Palette de couleurs pour les communautés
community_cmap = cm.get_cmap('tab20', len(communities))
community_colors = {comm: community_cmap(i) for i, comm in enumerate(communities)}

# Layout avec ajustement pour rapprocher les nœuds
# Utilisation de la force d'attraction entre communautés différentes
pos_initial = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
pos = pos_initial.copy()

# Ajustement du layout pour regrouper par communauté mais avec moins de séparation
for node in G.nodes():
    comm = partition[node]
    # Décalage plus faible pour réduire l'espacement entre communautés
    pos[node][0] += community_colors[comm][0] * 0.6
    pos[node][1] += community_colors[comm][1] * 0.6

# Palette de couleurs pour les gagnants
unique_gagnants = list(set(df_filtered['Gagnant'].unique()))
gagnant_cmap = cm.get_cmap('Set1', len(unique_gagnants))
gagnant_to_color = {g: gagnant_cmap(i) for i, g in enumerate(unique_gagnants)}

# Calculer les poids moyens des arêtes pour chaque communauté
community_edge_weights = defaultdict(list)
for u, v, data in G.edges(data=True):
    if partition[u] == partition[v]:  # Si les deux nœuds sont dans la même communauté
        community_edge_weights[partition[u]].append(data['weight'])

# Calculer la moyenne des poids pour chaque communauté
avg_weights = {comm: np.mean(weights) if weights else 0 
              for comm, weights in community_edge_weights.items()}

# Visualisation améliorée
plt.figure(figsize=(24, 18))
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# Dessiner les arêtes avec couleur par communauté
for u, v, data in G.edges(data=True):
    if partition[u] == partition[v]:  # Même communauté
        comm = partition[u]
        color = community_colors[comm]
        # Épaisseur proportionnelle au poids
        width = data['weight'] * 3
        alpha = min(0.8, data['weight'])
        plt.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 
                 color=color, linewidth=width, alpha=alpha)
    else:  # Communautés différentes
        # Arêtes intercommunautaires plus fines et transparentes
        plt.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 
                 color='lightgray', linewidth=0.5, alpha=0.2)

# Dessiner les nœuds avec taille en fonction du degré
node_sizes = [300 * (1 + G.degree(n) / 10) for n in G.nodes()]
node_colors = [gagnant_to_color[G.nodes[n]['gagnant']] for n in G.nodes()]

nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, 
                       edgecolors='black', linewidths=0.5, alpha=0.85)

# Sélectionner un sous-ensemble de nœuds pour les étiquettes (pour éviter la surcharge)
if len(G.nodes()) > 50:
    # Montrer les libellés uniquement pour les nœuds importants (ex: haut degré)
    degrees = dict(G.degree())
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:30]
    labels = {n: f"{n}\n(Comm. {partition[n]})" for n in top_nodes}
else:
    labels = {n: f"{n}\n(Comm. {partition[n]})" for n in G.nodes()}

nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight='bold', 
                        font_color='black', bbox=dict(facecolor='white', alpha=0.7, 
                                                   edgecolor='none', boxstyle='round,pad=0.2'))

# Création des éléments de légende pour les gagnants
legend_elements_gagnants = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=gagnant, 
         markersize=10, markeredgecolor='black', markeredgewidth=0.5)
    for gagnant, color in gagnant_to_color.items()
]

# Création des éléments de légende pour les poids moyens des arêtes par communauté
legend_elements_weights = []
for comm, avg_weight in avg_weights.items():
    legend_elements_weights.append(
        mpatches.Patch(color=community_colors[comm], 
                      label=f"Comm. {comm}: poids moy = {avg_weight:.3f}")
    )

# Création de deux zones de légendes séparées (deux axes)
# Légende des gagnants en haut à gauche
legend_gagnants = plt.legend(handles=legend_elements_gagnants, 
                           title="Gagnants", fontsize=10, 
                           loc="upper left", framealpha=0.9)

# Ajouter la première légende au plot
plt.gca().add_artist(legend_gagnants)

# Légende des poids en bas à droite
plt.legend(handles=legend_elements_weights, 
          title="Poids moyens par communauté", fontsize=10, 
          loc="lower right", framealpha=0.9)

# Titre et annotations
plt.title("Réseau des communes françaises par similarité pondérée", 
         fontsize=20, fontweight='bold', pad=20)
plt.figtext(0.5, 0.01, "Les liens entre communes de même communauté sont colorés selon la communauté. "
           "L'épaisseur des liens est proportionnelle à leur poids.", 
           ha='center', fontsize=12)

plt.axis('off')
plt.tight_layout()
plt.savefig("reseau_communes_communautes_ameliore.png", dpi=300, bbox_inches='tight')

# Pour visualiser les statistiques des communautés
community_stats = defaultdict(lambda: {'count': 0, 'gagnants': defaultdict(int)})
for node, comm in partition.items():
    community_stats[comm]['count'] += 1
    community_stats[comm]['gagnants'][G.nodes[node]['gagnant']] += 1

print("\nStatistiques des communautés:")
for comm, stats in sorted(community_stats.items()):
    print(f"\nCommunauté {comm}: {stats['count']} communes")
    print("Répartition des gagnants:")
    for gagnant, count in stats['gagnants'].items():
        print(f"  - {gagnant}: {count} ({count/stats['count']*100:.1f}%)")
    print(f"Poids moyen des liens: {avg_weights.get(comm, 0):.3f}")