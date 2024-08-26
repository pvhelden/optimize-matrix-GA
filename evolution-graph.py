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


# Create a directed graph
G = nx.DiGraph()

# Add nodes and edges to the graph
for _, row in data.iterrows():
    G.add_node(row['AC'], generation=row['generation'], AuROC=row['AuROC'])
    if row['parent_AC'] != "Origin":
        G.add_edge(row['parent_AC'], row['AC'])

# Create a layout for nodes: position them based on their generation and AuROC score
pos = {node: (data.loc[data['AC'] == node, 'generation'].values[0],
              data.loc[data['AC'] == node, 'AuROC'].values[0])
       for node in G.nodes()}

# Draw the plot
plt.figure(figsize=(12, 8))

# Draw edges manually to connect nodes with their parents
for node in G.nodes():
    for parent in G.predecessors(node):
        parent_pos = pos[parent]
        node_pos = pos[node]
        if parent_pos[0] < node_pos[0]:  # Ensure that the parent is from the previous generation
            plt.plot([parent_pos[0], node_pos[0]], [parent_pos[1], node_pos[1]], color='gray', linewidth=1)

# Draw nodes as small dots
x_values = [pos[node][0] for node in G.nodes()]
y_values = [pos[node][1] for node in G.nodes()]
plt.scatter(x_values, y_values, color='red', s=10)  # s is the size of the dots

# Set labels and title
plt.xlabel('Generation')
plt.ylabel('AuROC')
plt.title('Tree Structure Visualization')

# Show grid for better readability
plt.grid(True)

# Display the plot
plt.show()
