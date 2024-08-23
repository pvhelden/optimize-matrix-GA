import csv
import io
import math
import os
import random
import re
import subprocess

import polars as pl
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


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


def mutate_pssm(matrix, gen_nb=1, matrix_nb=1, n_children=None, min_percent=5, max_percent=25):
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


def compute_stats(pos_data, neg_data, score_col='weight', group_col='ft_name', score_threshold=-100):
    # Function to compute stats per group
    def compute_group_stats(df):
        # Sort by score in descending order
        df = df.sort(score_col, descending=True)

        # Add a row number column as rank
        df = df.with_columns(
            pl.arange(1, df.height + 1).alias("rank")
        )

        # Compute True Positives (TP) as the cumulative sum of label==1
        df = df.with_columns(pl.cum_sum('label').alias("TP"))

        # Compute False Positives (FP) as the cumulative sum of label==0
        df = df.with_columns((1 - pl.col('label')).cum_sum().alias("FP"))

        # Compute False Negative (FN) as the number of positives not yet counted
        pos_total = df["TP"].tail(1).to_list()[0]
        df = df.with_columns((pos_total - pl.col('TP')).alias("FN"))

        # Compute True Negative (TN) as the number of negatives not yet counted
        neg_total = df["FP"].tail(1).to_list()[0]
        df = df.with_columns((neg_total - pl.col('FP')).alias("TN"))

        # Compute True Positive Rate (recall, sensitivity)
        df = df.with_columns((pl.col("TP") / pos_total).alias("TPR"))

        # Compute False Positive Rate
        df = df.with_columns((pl.col("FP") / neg_total).alias("FPR"))

        # Compute Predictive positive Value (precision)
        df = df.with_columns((pl.col("TP") / (pl.col("TP") + pl.col("FP"))).alias("PPV"))

        # Add a column "new_value" indicating whether the score is higher in the current row than in the previous one
        df = df.with_columns(
            (
                # Check if current "weight" is greater than previous row's "weight"
                (pl.col(score_col) < pl.col(score_col).shift(1))
                .fill_null(True)  # Fill the first row's None with True
                .cast(pl.Int8)  # Cast the boolean result to integer (0 or 1)
                .alias("new_value")  # Name the column "new_value"
            )
        )

        # Compute cumulative AuROC
        df_new_score = df.filter(pl.col("new_value") == 1)
        df_new_score = df_new_score.with_columns(
            ((df_new_score["FPR"] - df_new_score["FPR"].shift(1)) * (
                    df_new_score["TPR"] + df_new_score["TPR"].shift(1)) / 2)
            .fill_null(0)  # Fill the first row's None with 0
            .cum_sum()
            .alias("AuROC_cum")
        )
        au_roc = df_new_score["AuROC_cum"].tail(1).to_list()[0]

        # Compute cumulative AuPR
        df_new_score = df.filter(pl.col("new_value") == 1)
        df_new_score = df_new_score.with_columns(
            ((df_new_score["TPR"] - df_new_score["TPR"].shift(1)) * (
                    df_new_score["PPV"] + df_new_score["PPV"].shift(1)) / 2)
            .fill_null(0)  # Fill the first row's None with 0
            .cum_sum()
            .alias("AuPR_cum")
        )

        au_pr = df_new_score["AuPR_cum"].tail(1).to_list()[0]

        scores, labels = df[score_col].to_numpy(), df["label"].to_numpy()

        # Compute ROC and AUC
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        # Compute Precision-Recall and AUC
        precision, recall, _ = precision_recall_curve(labels, scores)
        pr_auc = average_precision_score(labels, scores)

        return {
            "AuROC": au_roc,
            "roc_auc": roc_auc,
            "AuPR": au_pr,
            "pr_auc": pr_auc,
            "stat_per_score": df_new_score
        }

    # Merge positive and negative data frames
    data = pos_data.vstack(neg_data)

    # Filter out rows with score lower than threshold
    data = data.filter(pl.col(score_col) > score_threshold)

    # Process each group separately
    results = {}
    for group_value in data[group_col].unique():
        group_df = data.filter(pl.col(group_col) == group_value)  # Select the rows corresponding to current group_value
        group_stats = compute_group_stats(group_df)  # Compute performance stats on this group
        results[group_value] = group_stats  # Aggregate the results in a dictionary

    return results


def compute_stats_from_files(pos_file, neg_file, score_col='weight', group_col='ft_name', score_threshold=-100):
    # Read and preprocess the data
    def read_data(file, label):
        return pl.read_csv(
            file,
            comment_prefix=';',
            has_header=True,
            separator='\t',
        ).select([
            pl.col(score_col),
            pl.col(group_col),
            pl.lit(label).alias('label'),
        ]).filter(pl.col(score_col) >= score_threshold)

    # Load positive and negative datasets with labels
    pos_data = read_data(pos_file, 1)
    neg_data = read_data(neg_file, 0)
    result = compute_stats(pos_data=pos_data,
                           neg_data=neg_data,
                           score_col='weight',
                           group_col='ft_name',
                           score_threshold=-100)
    return result


def run_command(command, verbose=0):
    if verbose > 0:
        print('Running command:\n\t{}'.format(command))
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    return result
    # return {
    #    'command': command,
    #    'output': result.stdout,
    #    'error': result.stderr,
    #    'return_code': result.return_code
    # }


def scan_sequences(rsat_cmd, seq_file, label, matrix_file, bg_file,
                   score_col='weight', group_col='ft_name', score_threshold=-100):
    scan_cmd = (rsat_cmd +
                ' matrix-scan  -quick -v 1'
                ' -m ' + matrix_file +
                ' -matrix_format transfac'
                ' -i ' + seq_file +
                ' -seq_format fasta '
                '-bgfile ' + bg_file +
                ' -bg_pseudo 0.01 -pseudo 1 -decimals 1 -2str -return sites -uth rank_pm 1 -n score'
                )
    # Run the command
    result = run_command(scan_cmd)

    # Catch the stout in a buffer to enable post-processing with pl.read_csv
    # - filter out comment lines
    # - get the header
    # - select only required columns
    csv_buffer = io.StringIO(result.stdout)

    return pl.read_csv(
        csv_buffer,
        comment_prefix=';',
        has_header=True,
        separator='\t',
    ).select([
        pl.col(score_col),
        pl.col(group_col),
        pl.lit(label).alias('label'),
    ]).filter(pl.col(score_col) >= score_threshold)

    # return result


def score_matrix(rsat_cmd, seq_file_pos, seq_file_neg, matrix_file, bg_file):
    print("\tScoring matrix: " + matrix_file)
    print("\t\tScanning positive sequence file: " + seq_file_pos)
    pos_hits = scan_sequences(rsat_cmd=rsat_cmd, seq_file=seq_file_pos, label=1,
                              matrix_file=matrix_file, bg_file=bg_file)
    print("\t\tScanning negative sequence file: " + seq_file_neg)
    neg_hits = scan_sequences(rsat_cmd=rsat_cmd, seq_file=seq_file_neg, label=0,
                              matrix_file=matrix_file, bg_file=bg_file)
    print("\t\tComputing performance statistics (pos vs neg)")
    matrix_stat = compute_stats(pos_data=pos_hits, neg_data=neg_hits, score_col='weight', group_col='ft_name')
    return matrix_stat


def main():
    # ------------------------------------------------
    # Parameters
    # ------------------------------------------------
    min_percent = 5  # min percent change at each mutation
    max_percent = 30  # max percent change at each mutation
    nb_generations = 2  # number of generations
    # selection_size = 5  # number of individuals to keep from each generation
    n_children = 10  # number fo children per parent at each generation
    matrix_file = 'data/matrices/GABPA_CHS_THC_0866_peakmo-clust-trimmed.tf'
    # matrix_file = 'data/matrices/test_matrix_1.tf'
    scan_file_pos = 'data/scans/CHS_GABPA_THC_0866_peakmo-clust-matrices_train.tsv'
    scan_file_neg = 'data/scans/CHS_GABPA_THC_0866_peakmo-clust-matrices_rand.tsv'
    seq_file_pos = 'data/sequences/THC_0866.fasta'
    seq_file_neg = 'data/sequences/THC_0866_rand-loci_noN.fa'
    bg_file = 'data/bg_models/equiprobable_1str.tsv'
    rsat_version = '20240820'
    base_dir = '/Users/jvanheld/no_backup/rsat_github/optimize-matrix-GA'
    rsat_cmd = ('docker run -v '
                '{0}:/home/rsat_user '
                '-v {1}/results:/home/rsat_user/out '
                'eeadcsiccompbio/rsat:{2} rsat').format(
        base_dir, base_dir, rsat_version)

    # create directory to export matrices with performance scores
    matrix_out_dir = 'results/matrices'
    if not os.path.exists(matrix_out_dir):
        os.makedirs(matrix_out_dir)

    # ------------------------------------------------
    # Load original matrices
    # ------------------------------------------------
    print('\tLoading matrices from file: ' + matrix_file)
    matrices = parse_transfac(matrix_file)

    # print('\tExporting matrices to file: ' + matrix_file + '.test')
    # export_pssms(matrices, matrix_file + '.test')

    # ------------------------------------------------
    # Score matrices according to their capability to discriminate positive from negative sequences
    # ------------------------------------------------
    for matrix in matrices:
        # Export this matrix in a separate file for scanning
        matrix_ac = matrix['metadata']['AC']
        one_matrix_file = 'results/matrices/' + matrix_ac + '.tf'
        print('\t\tExporting matrix ' + matrix_ac + ' to file ' + one_matrix_file)
        export_pssms([matrix], one_matrix_file)

        # Compute performance statistics for this matrix
        matrix_stat = score_matrix(rsat_cmd, seq_file_pos, seq_file_neg, one_matrix_file, bg_file)

        # Append classification performance scores to matrix comments
        matrix['metadata']['CC'] = matrix['metadata']['CC'] + [
            '  Performance metrics',
            '    AuROC: ' + str(matrix_stat[matrix_ac]['AuROC']),
            '    AuPR: ' + str(matrix_stat[matrix_ac]['AuPR']),
        ]
        print('\t\tAuROC: ' + str(matrix_stat[matrix_ac]['AuROC']))
        print('\t\tAuPR: ' + str(matrix_stat[matrix_ac]['AuPR']))
        scored_matrix_file = 'results/matrices/' + matrix_ac + '_scored.tf'
        print('\t\tExporting scored matrix ' + matrix_ac + ' to file ' + scored_matrix_file)
        export_pssms([matrix], scored_matrix_file)

    # ------------------------------------------------
    # GA algorithm
    # ------------------------------------------------
    collected_matrices = matrices
    parent_matrices = matrices
    print('Matrix proliferation over ' + str(nb_generations) + ' generations; '
          + str(n_children), ' children per generation')

    # Iterate over generations
    for g in range(nb_generations):
        gen_nb = g + 1
        children_matrices = []
        print("Generation: " + str(gen_nb))
        print("\tparent matrices: " + str(len(parent_matrices)))
        for m in range(len(parent_matrices)):
            matrix = parent_matrices[m]
            # Collect mutated matrices
            mutated_matrices = mutate_pssm(
                matrix,
                gen_nb=gen_nb,
                matrix_nb=m + 1,
                n_children=n_children,
                min_percent=min_percent,
                max_percent=max_percent,
            )
            children_matrices = children_matrices + mutated_matrices
        print("\tchildren matrices: " + str(len(children_matrices)))
        collected_matrices = collected_matrices + children_matrices
        print("\tcollected matrices: " + str(len(collected_matrices)))

        parent_matrices = children_matrices

        # ------------------------------------------------
        # Export matrices collected at each generation
        # ------------------------------------------------
        outfile = "collected_matrices_gen" + str(gen_nb) + ".tf"
        print("\tExporting collected matrices to file\t" + outfile)
        export_pssms(collected_matrices, outfile)

    # ------------------------------------------------
    # Compute classification statistics per PSSM from two sequence scanning files (positive and negative data sets)
    # ------------------------------------------------
    print("\tComputing performance statistics with compute_stats_from_files()")
    stats_per_motif = compute_stats_from_files(scan_file_pos, scan_file_neg, 'weight', 'ft_name')
    print(stats_per_motif)

    # ------------------------------------------------
    # Convert the dictionary to a list of tuples and sort by AuROC in decreasing order
    # ------------------------------------------------
    stats_per_motif_sorted = sorted(stats_per_motif.items(), key=lambda x: x[1]['AuROC'], reverse=True)

    # Open the file in write mode
    with open("stats_per_motif.tsv", mode='w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')

        # Write the header
        writer.writerow(["key", "AuROC", "roc_auc", "AuPR", "pr_auc"])

        # Write the data
        for key, values in stats_per_motif_sorted:
            writer.writerow([
                key,
                f"{values['AuROC']:.4f}",
                f"{values['roc_auc']:.4f}",
                f"{values['AuPR']:.4f}",
                f"{values['pr_auc']:.4f}"
            ])


if __name__ == '__main__':
    main()
