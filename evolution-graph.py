import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# ----------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------

score_table = 'data/score_tables/PRDM5_GHTS_YWK_B_AffSeq_B1_PRDM5.C2_clust-trimmed-matrices_train-vs-rand_gen0-20_score_table.tsv'

# ----------------------------------------------------------------
# Load the data
# ----------------------------------------------------------------
data = pd.read_csv(score_table, sep='\t')

# ----------------------------------------------------------------
# Create a directed graph
# ----------------------------------------------------------------
G = nx.DiGraph()

# ----------------------------------------------------------------
# Add nodes and edges to the graph
# ----------------------------------------------------------------
for _, row in data.iterrows():
    G.add_node(row['AC'], generation=row['generation'], AuROC=row['AuROC'])
    if row['parent_AC'] != "Origin":
        G.add_edge(row['parent_AC'], row['AC'])

# ----------------------------------------------------------------
# Create a layout for nodes: position them based on their generation and AuROC score
# ----------------------------------------------------------------
pos = {node: (data.loc[data['AC'] == node, 'generation'].values[0],
              data.loc[data['AC'] == node, 'AuROC'].values[0])
       for node in G.nodes()}

# ----------------------------------------------------------------
# Draw the graph
# ----------------------------------------------------------------
plt.figure(figsize=(12, 8))
nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=8, font_weight='bold', edge_color='gray')

# ----------------------------------------------------------------
# Draw nodes with generation as x-axis and AuROC as y-axis
# ----------------------------------------------------------------
for node, (x, y) in pos.items():
    plt.scatter(x, y, color='red')
    plt.text(x, y, node, fontsize=8, ha='right', va='bottom')

# ----------------------------------------------------------------
# Set labels and title
# ----------------------------------------------------------------
plt.xlabel('Generation')
plt.ylabel('AuROC')
plt.title('Tree Structure Visualization')

plt.grid(True)
plt.show()
