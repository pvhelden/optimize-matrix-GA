import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import argparse
import os

def plot_tree(input_file, output_file, file_format, title, subtitle):
    # Load the data
    data = pd.read_csv(input_file, sep='\t')

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
    plt.title(title)
    if subtitle:
        plt.suptitle(subtitle, y=0.95, fontsize=10)

    # Set x-ticks to be integers and remove decimals
    plt.xticks(ticks=range(int(min(x_values)), int(max(x_values)) + 1))

    # Show grid for better readability
    plt.grid(True)

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save the plot to the specified file format
    plt.savefig(output_file, format=file_format)
    plt.close()

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Plot a tree structure from a TSV file and save it as an image.")

    parser.add_argument('-i', '--input_file', required=True, help="Path to the input TSV file.")
    parser.add_argument('-o', '--output_file', required=True, help="Full path of the output file.")
    parser.add_argument('-f', '--file_format', choices=['png', 'pdf'], required=True, help="Format of the output file. Supported formats: png, pdf.")
    parser.add_argument('-t', '--title', required=True, help="Title of the graph.")
    parser.add_argument('-s', '--subtitle', help="Subtitle of the graph.")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the plot function with the provided arguments
    plot_tree(args.input_file, args.output_file, args.file_format, args.title, args.subtitle)

if __name__ == '__main__':
    main()
