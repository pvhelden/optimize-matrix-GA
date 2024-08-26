import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import os


def plot_tree(input_file, output_file, file_format, title, subtitle, min_y=None, max_y=None, min_g=None, max_g=None,
              y_step1=0.1, y_step2=0.05, xsize=12, ysize=8, dpi=100):
    # Load the data
    data = pd.read_csv(input_file, sep='\t')

    # Filter data based on generation range if provided
    if min_g is not None:
        data = data[data['generation'] >= min_g]
    if max_g is not None:
        data = data[data['generation'] <= max_g]

    # Check if data is empty after filtering
    if data.empty:
        raise ValueError("Filtered data is empty. Check the generation range specified.")

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes and edges to the graph
    for _, row in data.iterrows():
        G.add_node(row['AC'], generation=row['generation'], AuROC=row['AuROC'])
        if row['parent_AC'] != "Origin":
            G.add_edge(row['parent_AC'], row['AC'])

    # Create a layout for nodes: position them based on their generation and AuROC score
    pos = {}
    for node in G.nodes():
        generation = data.loc[data['AC'] == node, 'generation']
        AuROC = data.loc[data['AC'] == node, 'AuROC']

        if not generation.empty and not AuROC.empty:
            pos[node] = (generation.values[0], AuROC.values[0])
        else:
            print(f"Warning: Node '{node}' is missing from the filtered data.")

    # Draw the plot
    plt.figure(figsize=(xsize, ysize))

    # Draw edges manually to connect nodes with their parents
    for node in G.nodes():
        for parent in G.predecessors(node):
            parent_pos = pos.get(parent)
            node_pos = pos.get(node)
            if parent_pos and node_pos:
                if parent_pos[0] < node_pos[0]:  # Ensure that the parent is from the previous generation
                    plt.plot([parent_pos[0], node_pos[0]], [parent_pos[1], node_pos[1]], color='#888888', linewidth=1)

    # Draw nodes as small dots
    x_values = [pos[node][0] for node in G.nodes() if node in pos]
    y_values = [pos[node][1] for node in G.nodes() if node in pos]
    plt.scatter(x_values, y_values, color='red', s=10)  # s is the size of the dots

    # Set labels and title
    plt.xlabel('Generation')
    plt.ylabel('AuROC')
    plt.title(title, pad=20)  # Main title

    if subtitle:
        # Place the subtitle directly below the title
        plt.text(0.5, 0.97, subtitle, ha='center', va='top', fontsize=10, transform=plt.gcf().transFigure)

    # Set x-ticks to be integers and remove decimals
    plt.xticks(ticks=range(int(min(x_values)), int(max(x_values)) + 1))

    # Set the Y-axis limits based on min_y and max_y
    if min_y is not None:
        plt.ylim(bottom=min_y)
    if max_y is not None:
        plt.ylim(top=max_y)

    # Set horizontal grid with different levels of steps
    y_ticks1 = [round(i * y_step1, 1) for i in range(int(min_y // y_step1), int(max_y // y_step1) + 1)]
    y_ticks2 = [round(i * y_step2, 2) for i in range(int(min_y // y_step2), int(max_y // y_step2) + 1)]

    plt.yticks(ticks=y_ticks1)
    plt.grid(True, which='major', axis='y', linestyle='-', color='#888888', linewidth=0.8)  # Mid-gray
    plt.grid(True, which='minor', axis='y', linestyle='--', color='#CCCCCC', linewidth=0.5)  # Pale gray

    # Set minor ticks for the second level of grid
    plt.gca().set_yticks(y_ticks2, minor=True)

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save the plot to the specified file format with optional DPI
    if file_format == 'png':
        plt.savefig(output_file, format=file_format, dpi=dpi)
    else:
        plt.savefig(output_file, format=file_format)

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description=("""
            Generates a plot of nodes from a tree-like data structure with their parent-child relationships.
            
            Input file should be a tab-separated text file including at least the following columns: 
            - generation (int)
            - AC (str): accession of the current node
            - AuROC (float): area under the receiving operating characteristic 
            - parent_AC (strr): AC of the parent node
            - selected (1 or 0): flag indicating whether or not the node is selected for next generation
            """
        )
    )

    parser.add_argument('-i', '--input_file', required=True,
                        help='Input file. TSV file with columns: generation, AC, AuROC, parent_AC.')
    parser.add_argument('-o', '--output_file', required=True, help='Output file path.')
    parser.add_argument('-f', '--file_format', choices=['png', 'pdf'], required=True,
                        help='Format of the output file. Supported: png, pdf.')
    parser.add_argument('-t', '--title', required=True, help='Title of the graph.')
    parser.add_argument('-s', '--subtitle', help='Subtitle of the graph.')
    parser.add_argument('--min_y', type=float, help='Minimum value for the Y axis.')
    parser.add_argument('--max_y', type=float, help='Maximum value for the Y axis.')
    parser.add_argument('--min_g', type=int, help='Minimum generation to include.')
    parser.add_argument('--max_g', type=int, help='Maximum generation to include.')
    parser.add_argument('--y_step1', type=float, default=0.1, help='Step size for the major grid lines on Y axis.')
    parser.add_argument('--y_step2', type=float, default=0.05, help='Step size for the minor grid lines on Y axis.')
    parser.add_argument('--xsize', type=float, default=12, help='Width of the output plot in inches.')
    parser.add_argument('--ysize', type=float, default=8, help='Height of the output plot in inches.')
    parser.add_argument('--dpi', type=int, default=100, help='Dots per inch (DPI) for PNG output.')
    parser.add_argument('-v', '--verbosity', type=int, default=0,
                        help='Verbosity level: if greater than 0, print the full command used.')

    args = parser.parse_args()

    if args.verbosity > 0:
        print(f"Running command: python plot-auroc-profiles.py "
              f"-i {args.input_file} "
              f"-o {args.output_file} "
              f"-f {args.file_format} "
              f"-t \"{args.title}\" "
              f"-s \"{args.subtitle}\" "
              f"{'--min_y ' + str(args.min_y) if args.min_y is not None else ''} "
              f"{'--max_y ' + str(args.max_y) if args.max_y is not None else ''} "
              f"{'--min_g ' + str(args.min_g) if args.min_g is not None else ''} "
              f"{'--max_g ' + str(args.max_g) if args.max_g is not None else ''} "
              f"--y_step1 {args.y_step1} "
              f"--y_step2 {args.y_step2} "
              f"--xsize {args.xsize} "
              f"--ysize {args.ysize} "
              f"--dpi {args.dpi}")

    plot_tree(
        args.input_file,
        args.output_file,
        args.file_format,
        args.title,
        args.subtitle,
        args.min_y,
        args.max_y,
        args.min_g,
        args.max_g,
        args.y_step1,
        args.y_step2,
        args.xsize,
        args.ysize,
        args.dpi
    )


if __name__ == '__main__':
    main()
