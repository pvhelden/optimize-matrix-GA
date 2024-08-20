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


if __name__ == '__main__':
    matrix_file = 'data/matrices/GABPA_CHS_THC_0866_peakmo-clust-trimmed.tf'
    parsed_matrices = parse_transfac(matrix_file)
    for matrix in parsed_matrices:
        print("Metadata:", matrix['metadata'])
        print("Matrix DataFrame:", matrix['matrix'])
