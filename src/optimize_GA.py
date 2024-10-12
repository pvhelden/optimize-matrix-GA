import copy

import polars as pl

from src.logging import log_message
from src.matrix_scoring import score_matrices
from src.matrix_util import export_pssms, clone_and_mutate_pssm


def genetic_algorithm(matrices, rsat_cmd, seq_file_pos, seq_file_neg, bg_file, output_prefix, tmp_dir,
                      generations=4, select=5, children=10, threads=4, selection_score="AuROC"):
    """
    Perform a genetic algorithm for optimizing Position-Specific Scoring Matrices (PSSMs).

    This algorithm evolves an initial set of PSSMs over a specified number of generations,
    selecting the top-performing matrices in each generation and generating new matrices
    through cloning followed by mutation. The matrices are scored based on their ability
    to discriminate between positive and negative sequence sets.

    Parameters:
    -----------
        matrices (list of dict): A list of matrices to be processed. Each matrix is a dictionary containing
            metadata and a data frame representing the matrix.
        rsat_cmd (str): command to run RSAT software suite
        seq_file_pos (str): path to the positive sequence file.
        seq_file_neg (str): path to the negative sequence file.
        bg_file (str): path to the background file.
        matrix_out_dir (str, optional): directory to save the matrix files at each generation.
        tmp_dir (str, optional): directory to save the temporary matrix files used for sequence scanning
            (automatically deleted after scanning)

    rsat_cmd : str
        The command or path to the RSAT utility used for scoring the matrices.
    seq_file_pos : str
        The file path to the positive sequences in FASTA format.
    seq_file_neg : str
        The file path to the negative sequences in FASTA format.
    bg_file : str
        The file path to the background model used for scoring.
    n_generations : int
        The number of generations to evolve the matrices.
    k : int
        The number of top-scoring matrices to select for reproduction in each generation.
    mutation_rate : float
        The probability of applying a mutation to a matrix.
    n_children : int
        The number of offspring (mutated matrices) to generate from each selected parent matrix.
    file_prefix : str
        The prefix for the output files. Matrices for each generation will be saved with this
        prefix, followed by `_gen#`, where `#` is the generation number, and the `.tf` suffix.

    Returns:
    --------
    list
        A list of the final set of optimized matrices after the last generation.

    Example:
    --------
        final_matrices = genetic_algorithm(
            file_path="input.tf",
            rsat_cmd="/path/to/rsat",
            seq_file_pos="positives.fasta",
            seq_file_neg="negatives.fasta",
            bg_file="background.bg",
            n_generations=10,
            k=5,
            mutation_rate=0.1,
            n_children=10,
            file_prefix="output")
    """
    # Step 1: Clone the initial matrices (Generation 0)
    current_generation = copy.deepcopy(matrices)

    # Define parent_AC and parent_ID attributes for the original matrices in order to have a value in the score tables
    for matrix in current_generation:
        matrix['metadata']['parent_AC'] = "Origin"
        matrix['metadata']['parent_ID'] = "Origin"

    # Prepare a DataFrame to collect the score tables
    score_table_all_generations = pl.DataFrame([])

    # Evolution Process
    for generation in range(generations + 1):
        log_message('info', 1, f"Generation {generation}")

        # Score the matrices of the current generation
        log_message('info', 2, f"scoring {len(current_generation)} matrices")
        matrix_scores = score_matrices(current_generation, rsat_cmd, seq_file_pos, seq_file_neg, bg_file,
                                       tmp_dir=tmp_dir, n_threads=threads)

        # Export all the scored matrices of the current generation
        output_file = f"{output_prefix}_gen{generation}_scored.tf"
        log_message("info", 2, f"Saving {len(current_generation)} scored matrices to {output_file}")
        export_pssms(current_generation, output_file)

        # Sort matrices by decreasing score
        sorted_scores = sorted(matrix_scores, key=lambda x: x[selection_score], reverse=True)

        # Convert the sorted list into a DataFrame for a tabular view
        sorted_score_table = pl.DataFrame(sorted_scores)

        # Add a column with the generation number and reorder columns to place it in first position
        sorted_score_table = sorted_score_table.with_columns(pl.lit(generation).alias('generation'))
        sorted_score_table = sorted_score_table.select(['generation'] + sorted_score_table.columns[:-1])

        # Select the top-k scoring matrices
        top_ac_values = sorted_score_table.head(select)['AC'].to_list()

        # Label with 1 the selected matrices, and with 0 the other ones in the sorted score table
        sorted_score_table = sorted_score_table.with_columns([
            pl.col("AC").is_in(top_ac_values).cast(pl.Int8).alias("selected")
        ])

        # Filter the current_generation to keep only the matrices with top-k 'AC' values
        top_matrices = [entry for entry in current_generation if entry['metadata']['AC'] in top_ac_values]

        # Export the k top-scoring scored matrices of the current generation
        output_file = f"{output_prefix}_gen{generation}_scored_{selection_score}_top{select}.tf"
        log_message("info", 2, f"Saving top {select} scored matrices to {output_file}")
        export_pssms(top_matrices, output_file, out_format='transfac')

        # # Export matrix scores to TSV
        matrix_scores_tsv = f"{output_prefix}_gen{generation}_score_table.tsv"
        log_message("info", 2, f"Saving score table to {matrix_scores_tsv}")
        sorted_score_table.write_csv(matrix_scores_tsv, separator='\t', include_header=True)

        # Append the DataFrame to the list
        # all_sorted_score_tables.append(sorted_score_table)

        # Create a DataFrame aggregating the score tables of all generations
        score_table_all_generations = pl.concat([score_table_all_generations, sorted_score_table])

        # Create the next generation
        if generation < generations:
            log_message("info", 2, "Cloning and mutating offspring matrices")
            # Prepare for the next iteration
            # Include top matrices to next generation because some of them might be better than their offsprint
            next_generation = []
            next_generation.extend(top_matrices)
            for i, matrix in enumerate(top_matrices):
                mutated_matrices = clone_and_mutate_pssm(
                    matrix, gen_nb=generation + 1, matrix_nb=i + 1, n_children=children)
                next_generation.extend(mutated_matrices)
            current_generation = next_generation

    # Return the final set of optimized matrices + score table for all the generations
    return current_generation, score_table_all_generations
