import math
import random
import re

import numpy as np
import polars as pl


def validate_transfac(file_path):
    """
    Validate if the file is in TRANSFAC format.

    Parameters:
    -----------
    file_path : str
        The path to the file that needs to be validated.

    Returns:
    --------
    bool
        Returns True if the file is in the correct TRANSFAC format, otherwise False.

    Notes:
    ------
    This is a basic check and might need to be extended for full compliance.
    This is ChatGPT-generated code.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        # TRANSFAC typically starts with a 'ID' or 'AC' line
        if not lines or not (lines[0].startswith('ID') or lines[0].startswith('AC')):
            return False
    return True


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
                    matrix_df = pl.DataFrame(matrix_data, schema=['Position', 'A', 'C', 'G', 'T'], orient="row")
                    matrices.append({'metadata': current_metadata, 'matrix': matrix_df})
                    matrix_data = []
                current_metadata = {'AC': line.split('  ')[-1], 'CC': []}
                in_matrix = False
            elif line.startswith('ID'):
                current_metadata['ID'] = line.split('  ')[-1]
            elif line.startswith('DE'):
                current_metadata['DE'] = line.split('  ')[-1]
            elif line.startswith('CC'):
                # comment lines can be multiple
                current_metadata['CC'].append(line.split('  ')[-1])
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
            matrix_df = pl.DataFrame(matrix_data, schema=['Position', 'A', 'C', 'G', 'T'], orient="row")
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

    file_path : str | LiteralString
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
                f.write(f"{row['Position']:<4d}  {row['A']:8d}  {row['C']:8d}  {row['G']:8d}  {row['T']:8d}\n")

            f.write("XX\n")
            matrix_comments = metadata.get('CC', '')
            for comment in matrix_comments:
                f.write(f"CC  {comment}\n")
            f.write("//\n")


def rescale_to_target(numbers, target):
    """
    Rescale a list of numbers (integers or floats) so that their sum equals the specified target integer.
    The output list will contain only integers.

    Parameters:
    - numbers (list of int/float): The list of numbers to rescale.
    - target (int): The target sum for the rescaled numbers.

    Returns:
    - list of int: A list of integers rescaled from the input numbers summing to the target.
    """
    if not numbers:
        return []

    # Calculate the sum of the original numbers
    total_sum = sum(numbers)

    # Avoid division by zero if total_sum happens to be zero (when all elements are zero)
    if total_sum == 0:
        # Distribute the target value equally, as far as possible
        n = len(numbers)
        result = [target // n] * n
        remainder = target % n
        for i in range(remainder):
            result[i] += 1
        return result

    # Calculate scale factor
    scale_factor = target / total_sum

    # Scale and round numbers
    scaled_numbers = [x * scale_factor for x in numbers]
    rounded_numbers = [round(x) for x in scaled_numbers]

    # Calculate the rounding error
    rounded_sum = sum(rounded_numbers)
    error = target - rounded_sum

    # If there's an error, distribute it (positive or negative error)
    if error != 0:
        # Sort indices of numbers by the size of their fractional parts
        fractions = [(i, x - int(x)) for i, x in enumerate(scaled_numbers)]
        # Correct positive or negative discrepancies
        correction_indices = sorted(fractions, key=lambda x: -abs(x[1] - 0.5) if error < 0 else abs(x[1] - 0.5))

        for i in range(abs(error)):
            idx = correction_indices[i][0]
            rounded_numbers[idx] += 1 if error > 0 else -1

    return rounded_numbers


def apply_mutation(original_counts: list[int], residue_index: int, percent_change: float):
    """
    Applies a mutation to a specific position in a PSSM.

    Parameters
    ----------
    original_counts : List[int]
        A list of counts for the residues [A, C, G, T] at the specified position.

    residue_index : int
        The index of the residue to be mutated in the PSSM column.

    percent_change : float
        The percentage by which to increase the count of the selected residue.

    Returns
    -------
    List[int]
        A list of the mutated counts for the residues [A, C, G, T] at the specified position.
    """
    total_counts = sum(original_counts)
    mutated_counts = original_counts.copy()

    change_amount = min(round(total_counts * (percent_change / 100)), total_counts - original_counts[residue_index])

    if change_amount > 0:
        mutated_counts[residue_index] += change_amount

        # Adjust other residues proportionally
        other_residue_sum = total_counts - original_counts[residue_index]
        for i in range(4):
            if i != residue_index:
                adj = math.ceil(original_counts[i] * (change_amount / other_residue_sum))
                mutated_counts[i] -= adj

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

    if sum(mutated_counts) != total_counts:
        mutated_counts = rescale_to_target(mutated_counts, total_counts)

    return mutated_counts


def clone_and_mutate_pssm(matrix, gen_nb=1, matrix_nb=1, n_children=None, min_percent=5, max_percent=25):
    """
    Generates a set of mutated position-specific scoring matrices (PSSMs) from a parent matrix by randomly mutating one
    position per child matrix.

    Parameters
    ----------
    matrix : Dict[str, Union[Dict[str, str], pl.DataFrame]]
        The input PSSM, structured as a dictionary containing 'metadata' and 'matrix' (a Polars DataFrame).

    gen_nb : int, optional
        The generation number to track the iteration of mutations. Default is 1.

    matrix_nb : int, optional
        Number of the matrix (in this generation) to track the iteration of mutations. Default is 1.

    n_children : int, optional
        The number of children matrices to generate. If not specified, it defaults to the number of positions in the
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
    height = matrix['matrix'].height
    if n_children is None:
        n_children = height  # Default to number of positions

    parent_ac = matrix['metadata']['AC']
    parent_id = matrix['metadata']['ID']
    parent_cc = matrix['metadata']['CC']

    # Remove any existing _G#_D# suffix
    ac_prefix = re.sub(r'_G\d+_M\d+_C\d+$', '', parent_ac)
    id_prefix = re.sub(r'_G\d+_M\d+_C\d+$', '', matrix['metadata']['ID'])

    mutated_matrices = []

    for child_nb in range(1, n_children + 1):
        # Clone the original matrix for each child
        mutated_matrix_df = matrix['matrix'].clone()

        # Randomly select one position to mutate
        mutated_position_index = random.randint(0, height - 1)
        original_counts = list(mutated_matrix_df.row(mutated_position_index))[1:]

        # Select a residue to mutate
        residue_index = random.randint(0, 3)
        # residue_index = 3 # JvH tmp

        # Select percent change randomly
        percent_change = random.uniform(min_percent, max_percent)

        # Apply mutation using the apply_mutation function
        mutated_counts = apply_mutation(original_counts, residue_index, percent_change)

        # Update the row with mutated counts
        mutated_row = [mutated_position_index + 1] + mutated_counts
        for index, item in enumerate(mutated_row):
            mutated_matrix_df[mutated_position_index, index] = item

        # Update metadata
        mutated_metadata = matrix['metadata'].copy()
        mutated_metadata['AC'] = f"{ac_prefix}_G{gen_nb}_M{matrix_nb}_C{child_nb}"
        mutated_metadata['ID'] = f"{id_prefix}_G{gen_nb}_M{matrix_nb}_C{child_nb}"
        mutated_metadata['parent_AC'] = parent_ac
        mutated_metadata['parent_ID'] = parent_id
        pcm_np = mutated_matrix_df.to_numpy()
        pcm_np = pcm_np[:, 1:]
        mutated_metadata['DE'] = build_consensus(pcm_np)
        mutated_metadata['CC'] = parent_cc + [
            f"Generation number: {gen_nb}",
            f"  Parent matrix number: {matrix_nb}",
            f"  AC of parent matrix: {parent_ac}",
            f"  ID of parent matrix: {parent_id}",
            f"  Child number: {child_nb}",
            f"  Mutated position {mutated_position_index + 1}, " +
            f"residue number {residue_index + 1}, " +
            f"percent change {percent_change:.2f}%",
        ]

        mutated_matrices.append({
            'metadata': mutated_metadata,
            'matrix': mutated_matrix_df
        })

    return mutated_matrices


def build_pcm(sequences):
    """
    Build a Position Count Matrix (PCM) from a set of aligned sequences.

    This function generates a position count matrix where each row corresponds
    to a position in the alignment and each column corresponds to one of the
    nucleotides (A, C, G, T). The matrix contains the counts of each nucleotide
    at each position.

    Parameters:
    ----------
    sequences : list of str
        A list of aligned nucleotide sequences of equal length.

    Returns:
    -------
    pcm : numpy.ndarray
        A 2D numpy array where rows correspond to positions in the alignment
        and columns correspond to nucleotides (A, C, G, T). The values in the
        matrix represent the count of each nucleotide at each position.

    Example:
    -------
    sequences = ["ATGCGT", "ATGAGT", "ATGCGT", "ATGCGC"]
    pcm = build_pcm(sequences)
    print(pcm)
    [[4. 0. 0. 0.]
     [0. 0. 0. 4.]
     [0. 0. 4. 0.]
     [0. 2. 2. 0.]
     [0. 0. 4. 0.]
     [0. 1. 0. 3.]]

    This code was generated by ChatGPT.
    """
    # Determine the length of the sequences
    positions = len(sequences[0])

    # Initialize the PCM with zeros
    pcm = np.zeros((positions, 4))  # A=0, C=1, G=2, T=3

    # Mapping nucleotides to columns
    nucleotide_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    # Fill the PCM with counts
    for seq in sequences:
        for i, nucleotide in enumerate(seq):
            pcm[i, nucleotide_index[nucleotide]] += 1

    return pcm


def build_consensus(pcm, prior=None, pseudocount=1):
    """
    Build a degenerate consensus sequence using a Position Count Matrix (PCM) and prior probabilities.

    This function computes a weight matrix based on the log-likelihood between
    the observed residue frequency in the PCM and the prior probability of each
    residue. It then generates a consensus sequence using IUPAC codes to
    represent positions where multiple nucleotides have positive weights.
    The consensus letter is uppercase if at least one residue has a weight
    greater than or equal to 1; otherwise, it is lowercase.

    Parameters:
    ----------
    pcm : numpy.ndarray or polars.DataFrame
        A 2D numpy array or Polars DataFrame where rows correspond to positions
        in the alignment and columns correspond to nucleotides (A, C, G, T).
        The values in the matrix represent the count of each nucleotide at each position.

    prior : dict, optional
        A dictionary representing the prior probabilities of each nucleotide.
        If None, an equiprobable prior is used (default: {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25}).

    pseudocount : int
        A small positive integer added to the counts to avoid division by zero.

    Returns:
    -------
    consensus : str
        A consensus sequence with IUPAC codes representing ambiguous positions
        where multiple nucleotides have positive weights. The letter is uppercase
        if at least one residue has a weight >= 1, otherwise lowercase.

    Example:
    -------
    pcm = pl.DataFrame({
            "A": [4, 0, 0, 0, 0, 0],
            "C": [0, 0, 0, 2, 0, 1],
            "G": [0, 0, 4, 2, 4, 0],
            "T": [0, 4, 0, 0, 0, 3]
        })
    consensus = build_consensus(pcm)
    print(consensus)
    'atGcgk'

    This code was generated by ChatGPT.
    """

    # Set default prior if None is provided
    if prior is None:
        prior = {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25}

    # Convert Polars DataFrame to NumPy array if necessary
    if isinstance(pcm, pl.DataFrame):
        pcm = pcm.to_numpy()

    # Ensure PCM has only 4 columns (A, C, G, T)
    if pcm.shape[1] > 4:
        pcm = pcm[:, 1:5]  # Keep only columns 2 to 5

    # Add pseudocount to PCM
    pcm_with_pseudocount = pcm + pseudocount

    # Number of positions
    positions = pcm_with_pseudocount.shape[0]

    # Calculate frequencies from the PCM with pseudocount
    frequencies = pcm_with_pseudocount / pcm_with_pseudocount.sum(axis=1, keepdims=True)

    # Compute the weight matrix
    prior_probs = np.array([prior['A'], prior['C'], prior['G'], prior['T']])
    weight_matrix = np.log2(frequencies / prior_probs)

    # IUPAC code mapping
    nucleotide_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    iupac_code = {
        frozenset(['A']): 'A', frozenset(['C']): 'C', frozenset(['G']): 'G', frozenset(['T']): 'T',
        frozenset(['A', 'G']): 'R', frozenset(['C', 'T']): 'Y', frozenset(['G', 'C']): 'S',
        frozenset(['A', 'T']): 'W', frozenset(['G', 'T']): 'K', frozenset(['A', 'C']): 'M',
        frozenset(['C', 'G', 'T']): 'B', frozenset(['A', 'G', 'T']): 'D', frozenset(['A', 'C', 'T']): 'H',
        frozenset(['A', 'C', 'G']): 'V', frozenset(['A', 'C', 'G', 'T']): 'N'
    }

    # Generate the consensus sequence with IUPAC codes
    consensus = []
    for i in range(positions):
        # Find nucleotides with positive weights
        positive_nucleotides = [nucleotide for nucleotide, idx in nucleotide_index.items() if weight_matrix[i, idx] > 0]

        # Determine the IUPAC code for the position
        iupac_letter = iupac_code[frozenset(positive_nucleotides)]

        # Check if any weight >= 1 to determine case
        if any(weight_matrix[i, idx] >= 1 for nucleotide, idx in nucleotide_index.items() if
               nucleotide in positive_nucleotides):
            consensus.append(iupac_letter.upper())
        else:
            consensus.append(iupac_letter.lower())

    return ''.join(consensus)


def pcm_to_pwm(pcm_df, pseudocount=1, priors=None):
    """
    Converts a Polars DataFrame representing a Position Count Matrix (PCM) into a Position Weight Matrix (PWM).

    Parameters:
    pcm_df (pl.DataFrame): A Polars DataFrame representing the PCM, with columns ['A', 'C', 'G', 'T'].
    pseudocount (float): A small value added to the counts to avoid log(0). Default is 1.
    background_freq (dict): A dictionary representing background frequencies of nucleotides (A, C, G, T).
                            If None, a uniform distribution of 0.25 for each nucleotide is assumed.

    Returns:
    pl.DataFrame: A Polars DataFrame representing the PWM with the same structure as the PCM.
    """
    # Ensure the PCM DataFrame contains the necessary columns
    required_columns = ['A', 'C', 'G', 'T']
    if not all(col in pcm_df.columns for col in required_columns):
        raise ValueError(f"The PCM DataFrame must contain the following columns: {required_columns}")

    # Set default background frequencies if not provided (uniform distribution)
    if priors is None:
        priors = {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25}

    # Convert the PCM into PWM
    pwm_data = []
    for row in pcm_df.iter_rows(named=True):
        total_counts = sum(row.values())
        pwm_row = {}
        for nucleotide, count in row.items():
            if nucleotide not in required_columns:
                continue
            # Calculate the frequency with pseudocount
            frequency = (count + pseudocount * priors[nucleotide]) / (total_counts + pseudocount)
            # Calculate log-odds ratio for the PWM
            pwm_value = np.log(frequency / priors[nucleotide])
            pwm_row[nucleotide] = pwm_value
        pwm_data.append(pwm_row)

    # Convert the PWM data back into a Polars DataFrame
    pwm_df = pl.DataFrame(pwm_data)

    return pwm_df
