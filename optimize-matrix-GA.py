import random
import re

import polars as pl


def parse_transfac(file_path):
    """
    Parses a TRANSFAC format file containing one or more matrices and returns a list of parsed matrices.

    Each matrix is stored as a dictionary containing the metadata and the matrix data itself. The metadata
    includes information like the accession number (AC), identifier (ID), and description (DE). The matrix
    data is stored as a Polars DataFrame, with columns representing the positions and nucleotide frequencies
    (A, C, G, T) at each position.

    Parameters
    ----------
    file_path : str
        The path to the TRANSFAC format file to be parsed.

    Returns
    -------
    List[Dict[str, Union[Dict[str, str], pl.DataFrame]]]
        A list of dictionaries where each dictionary represents a matrix. Each dictionary contains:
        - 'metadata': A dictionary with keys 'AC', 'ID', and 'DE', containing the accession number, identifier,
                      and description of the matrix, respectively.
        - 'matrix': A Polars DataFrame with columns 'Position', 'A', 'C', 'G', and 'T', representing the
                    nucleotide frequencies at each position in the matrix.

    Example
    -------
    parsed_matrices = parse_transfac("path_to_your_transfac_file.txt")
    for matrix_data in parsed_matrices:
        print("Metadata:", matrix_data['metadata'])
        print("Matrix DataFrame:", matrix_data['matrix'])

    Notes
    -----
    - The function assumes the TRANSFAC file contains matrices separated by `//` and that each matrix block
      starts with an `AC` (Accession number) line.
    - The `P0` line is considered the header for the matrix data, indicating the start of the nucleotide counts.
    - This function is designed to handle files with multiple matrices, storing each matrix separately in
      the output list.
    """
    matrices = []
    current_metadata = {}
    matrix_data = []
    in_matrix = False

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Ignore empty lines
            if not line:
                continue

            # Check for the beginning of a new matrix (AC or ID indicates a new matrix)
            if line.startswith('AC'):
                if in_matrix:  # If we're in the middle of a matrix, store the last one
                    matrix_df = pl.DataFrame(matrix_data, schema=['Position', 'A', 'C', 'G', 'T'])
                    matrices.append({'metadata': current_metadata, 'matrix': matrix_df})
                    matrix_data = []
                current_metadata = {'AC': line.split('  ')[-1]}
                in_matrix = False
            elif line.startswith('ID'):
                current_metadata['ID'] = line.split('  ')[-1]
            elif line.startswith('DE'):
                current_metadata['DE'] = line.split('  ')[-1]
            elif line.startswith('P0'):
                in_matrix = True
            elif in_matrix:
                if line[0].isdigit():
                    parts = line.split()
                    position = int(parts[0])
                    counts = list(map(int, parts[1:]))
                    matrix_data.append((position, *counts))
            elif line.startswith('//'):  # End of the current matrix
                if in_matrix:
                    matrix_df = pl.DataFrame(matrix_data, schema=['Position', 'A', 'C', 'G', 'T'])
                    matrices.append({'metadata': current_metadata, 'matrix': matrix_df})
                    matrix_data = []
                current_metadata = {}
                in_matrix = False

        # If the file doesn't end with '//' but there's an ongoing matrix, save it
        if matrix_data:
            matrix_df = pl.DataFrame(matrix_data, schema=['Position', 'A', 'C', 'G', 'T'])
            matrices.append({'metadata': current_metadata, 'matrix': matrix_df})

    return matrices


def export_pssms(pssms, file_path, out_format='transfac'):
    """
    Exports one or several position-specific scoring matrices (PSSM) in a given format.

    The PSSMs are provided as a list of dictionaries, where each dictionary contains metadata and
    a Polars DataFrame representing the matrix. By default, the function exports the matrices in
    TRANSFAC format.

    Parameters
    ----------
    pssms : List[Dict[str, Union[Dict[str, str], pl.DataFrame]]]
        A list of dictionaries where each dictionary represents a PSSM. Each dictionary should contain:
        - 'metadata': A dictionary with keys such as 'AC', 'ID', and 'DE', containing the accession number,
                      identifier, and description of the matrix, respectively.
        - 'matrix': A Polars DataFrame with columns 'Position', 'A', 'C', 'G', and 'T', representing the
                    nucleotide frequencies at each position in the matrix.

    file_path : str
        The path to the file where the exported matrices will be saved.

    out_format : str, optional
        The format to export the matrices. Currently supported: 'transfac'. Default is 'transfac'.

    Returns
    -------
    None
        The function writes the exported matrices to the specified file.

    Example
    -------
    export_pssms(parsed_matrices, "exported_pssms.txt")

    Notes
    -----
    - The default format is 'transfac'. If other formats are needed, the function can be extended to support them.
    - The function assumes that the input PSSMs are provided in the same structure as the output of the
      `parse_transfac` function.
    """
    if out_format.lower() != 'transfac':
        raise ValueError("Currently, only 'transfac' format is supported.")

    with open(file_path, 'w') as f:
        for pssm in pssms:
            metadata = pssm['metadata']
            matrix_df = pssm['matrix']

            # Write metadata
            f.write(f"AC  {metadata.get('AC', '')}\n")
            f.write("XX\n")
            f.write(f"ID  {metadata.get('ID', '')}\n")
            f.write("XX\n")
            f.write(f"DE  {metadata.get('DE', '')}\n")
            f.write("P0           a         c         g         t\n")

            # Write matrix
            for row in matrix_df.iter_rows(named=True):
                f.write(f"{row['Position']:2d}  {row['A']:10d}  {row['C']:10d}  {row['G']:10d}  {row['T']:10d}\n")

            f.write("XX\n")
            f.write("//\n")
            f.write("\n")  # Separate matrices with a blank line


def apply_mutation(original_counts, percent_change):
    """
    Applies a mutation to a specific position in a PSSM.

    Parameters
    ----------
    original_counts : List[int]
        A list of counts for the residues [A, C, G, T] at the specified position.

    percent_change : float
        The percentage by which to increase the count of the selected residue.

    Returns
    -------
    List[int]
        A list of the mutated counts for the residues [A, C, G, T] at the specified position.
    """
    total_counts = sum(original_counts)
    mutated_counts = original_counts.copy()

    # Select a residue to mutate
    residue_index = random.randint(0, 3)
    change_amount = round(original_counts[residue_index] * (percent_change / 100))
    mutated_counts[residue_index] += change_amount

    # Adjust other residues proportionally
    remaining_change = change_amount
    for i in range(4):
        if i != residue_index:
            adjustment = round(original_counts[i] * (remaining_change / total_counts))
            mutated_counts[i] -= adjustment

    # Ensure no counts fall below zero (compensate from others if necessary)
    for i in range(4):
        if mutated_counts[i] < 0:
            diff = -mutated_counts[i]
            mutated_counts[i] = 0
            # Distribute the difference proportionally to the others
            for j in range(4):
                if j != i and mutated_counts[j] > 0:
                    mutated_counts[j] += round(
                        diff * (original_counts[j] / sum([original_counts[k] for k in range(4) if k != i])))

    return mutated_counts


def mutate_pssm(matrix, gen_nb=1, n_desc=None, min_percent=5, max_percent=10):
    """
    Generates a set of mutated position-specific scoring matrices (PSSMs) from an input matrix by randomly mutating one
    position per descendant matrix.

    Parameters
    ----------
    matrix : Dict[str, Union[Dict[str, str], pl.DataFrame]]
        The input PSSM, structured as a dictionary containing 'metadata' and 'matrix' (a Polars DataFrame).

    gen_nb : int, optional
        The generation number to track the iteration of mutations. Default is 1.

    n_desc : int, optional
        The number of descendant matrices to generate. If not specified, it defaults to the number of positions in the
        matrix.

    min_percent : float, optional
        The minimum percentage change for mutations. Default is 5%.

    max_percent : float, optional
        The maximum percentage change for mutations. Default is 10%.

    Returns
    -------
    List[Dict[str, Union[Dict[str, str], pl.DataFrame]]]
        A list of mutated PSSMs, each structured as a dictionary containing 'metadata' and 'matrix'.
    """
    if n_desc is None:
        n_desc = matrix['matrix'].shape[0]  # Default to number of positions

    original_ac = matrix['metadata']['AC']
    # Remove any existing _G#_D# suffix
    original_ac = re.sub(r'_G\d+_D\d+$', '', original_ac)

    mutated_matrices = []

    for desc_num in range(1, n_desc + 1):
        # Clone the original matrix for each descendant
        mutated_matrix_df = matrix['matrix'].clone()

        # Randomly select one position to mutate
        random_position_index = random.randint(0, mutated_matrix_df.shape[0] - 1)
        selected_row = mutated_matrix_df.slice(random_position_index, 1).to_pandas().iloc[
            0]  # Get a single row as a Pandas DataFrame
        position_to_mutate = selected_row['Position']
        original_counts = [selected_row['A'], selected_row['C'], selected_row['G'], selected_row['T']]

        percent_change = random.uniform(min_percent, max_percent)
        # Apply mutation using the apply_mutation function
        mutated_counts = apply_mutation(original_counts, percent_change)

        # Update the row with mutated counts
        updates = {col: pl.when(pl.col("Position") == position_to_mutate).then(mutated_counts[i]).otherwise(pl.col(col))
                   for i, col in enumerate(['A', 'C', 'G', 'T'])}
        mutated_matrix_df = mutated_matrix_df.with_columns(list(updates.values()))

        # Update metadata
        mutated_metadata = matrix['metadata'].copy()
        mutated_metadata['AC'] = f"{original_ac}_G{gen_nb}_D{desc_num}"
        mutated_metadata['CC'] = [
            f"AC of original matrix: {original_ac}",
            f"Generation number: {gen_nb}",
            f"Mutated position {position_to_mutate}, percent change {percent_change:.2f}%"
        ]

        mutated_matrices.append({
            'metadata': mutated_metadata,
            'matrix': mutated_matrix_df
        })

    return mutated_matrices


def main():
    matrix_file = 'data/matrices/GABPA_CHS_THC_0866_peakmo-clust-trimmed.tf'
    parsed_matrices = parse_transfac(matrix_file)
    # for matrix in parsed_matrices:
    #     print("Metadata:", matrix['metadata'])
    #     print("Matrix DataFrame:", matrix['matrix'])

    # export_pssms(parsed_matrices, "exported_pssms.txt")

    test_matrix = parsed_matrices[1]
    mutated_matrices = mutate_pssm(test_matrix)
    export_pssms(mutated_matrices, "exported_pssms.txt")


if __name__ == '__main__':
    main()
