import concurrent.futures
import os

import polars as pl
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from src.logging import log_message
from src.matrix_util import export_pssms
from src.scan_sequences_rsat import scan_sequences_rsat


def score_matrix(matrix, rsat_cmd, seq_file_pos, seq_file_neg, bg_file, tmp_dir='tmp'):
    """
    Processes a single matrix by performing the following steps:
    1. Exports the matrix to a file for scanning.
    2. Computes performance statistics for the matrix.
    3. Appends classification performance scores to the matrix metadata.
    4. Exports the scored matrix to another file.

    Args:
        matrix (dict): A dictionary representing the matrix, including its metadata.
        rsat_cmd (str): command to run RSAT software suite
        seq_file_pos (str): path to the positive sequence file.
        seq_file_neg (str): path to the negative sequence file.
        bg_file (str): path to the background file.
        tmp_dir (str, optional): temporary directory to save the single-matrix files used by matrix-scan-quick.
            These files are automatically deleted after use.

    Side Effects:
        - Writes matrix data to disk.
        - Prints performance metrics to the console.
    """
    # Export this matrix in a separate file for scanning
    matrix_ac = matrix['metadata']['AC']
    log_message("info", 3, f"Scoring matrix {matrix_ac}")

    # single_matrix_file = os.path.join(tmp_dir, matrix_ac + '.tf')
    single_matrix_file = tmp_dir + '/' + matrix_ac + '.tf'
    log_message("info", 4, f"Exporting matrix {matrix_ac} to file {single_matrix_file}")
    export_pssms([matrix], single_matrix_file)

    # Compute performance statistics for this matrix
    # matrix_stat = score_matrix(rsat_cmd, seq_file_pos, seq_file_neg, single_matrix_file, bg_file)
    log_message("info", 4, f"Scanning positive sequence file: {seq_file_pos}")
    pos_hits = scan_sequences_rsat(rsat_cmd=rsat_cmd, seq_file=seq_file_pos, label=1,
                                   matrix_file=single_matrix_file, bg_file=bg_file)
    log_message("info", 4, "Scanning negative sequence file: " + seq_file_neg)
    neg_hits = scan_sequences_rsat(rsat_cmd=rsat_cmd, seq_file=seq_file_neg, label=0,
                                   matrix_file=single_matrix_file, bg_file=bg_file)
    log_message("info", 4, "Computing performance statistics (pos vs neg)")
    matrix_stat = compute_stats(pos_data=pos_hits, neg_data=neg_hits, score_col='weight', group_col='ft_name')

    # Append classification performance scores to matrix comments
    matrix['metadata']['CC'] = matrix['metadata']['CC'] + [
        '  Performance metrics',
        '    AuROC: ' + str(matrix_stat[matrix_ac]['AuROC']),
        '    AuPR: ' + str(matrix_stat[matrix_ac]['AuPR']),
    ]
    log_message("info", 4, 'AuROC: ' + str(matrix_stat[matrix_ac]['AuROC']))
    log_message("info", 4, 'AuPR: ' + str(matrix_stat[matrix_ac]['AuPR']))

    # Remove the temporary file used for scanning with matrix-scan-quick
    os.remove(single_matrix_file)

    # Return matrix and the associated statistics
    result = {
        "AC": matrix['metadata']['AC'],
        "ID": matrix['metadata']['ID'],
        "AuROC": matrix_stat[matrix_ac]['AuROC'],
        "roc_auc": matrix_stat[matrix_ac]['roc_auc'],
        "AuPR": matrix_stat[matrix_ac]['AuPR'],
        "pr_auc": matrix_stat[matrix_ac]['pr_auc'],
        "parent_AC": matrix['metadata'].get('parent_AC'),  # returns parent_AC or None for the root matrices
        "parent_ID": matrix['metadata'].get('parent_ID'),  # returns parent_ID or None for the root matrices
    }
    return result


def score_matrices(matrices, rsat_cmd, seq_file_pos, seq_file_neg, bg_file,
                   tmp_dir='tmp', n_threads=4):
    """
    Code generated by ChatGPT.

    Parallelizes the processing of multiple matrices using a thread pool.

    Args:
        matrices (list of dict): A list of matrices to be processed. Each matrix is a dictionary containing
            metadata and a data frame representing the matrix.
        rsat_cmd (str): command to run RSAT software suite
        seq_file_pos (str): path to the positive sequence file.
        seq_file_neg (str): path to the negative sequence file.
        bg_file (str): path to the background file.
        tmp_dir (str, optional): directory to save temporarily the single-matrix files used by matrix-scan-quick.
        n_threads (int, optional): The number of worker threads to use for parallel processing. Default is 4.

    Returns:
        dict: A dictionary where each key is the accession number of a matrix, and the value is the result dictionary
              returned by `process_matrix`.

    Side Effects:
        - Executes the `process_matrix` function concurrently on each matrix in the list.
        - Writes matrix data to disk.
        - Prints performance metrics to the console.

    Raises:
        Exception: If any exceptions are raised during the execution of threads, they will be re-raised.

    Example usage:
        parallel_process_matrices(
            matrices, rsat_cmd, seq_file_pos, seq_file_neg, bg_file, tmp_dir, n_threads=4)
    """
    # Create a thread pool with the desired number of workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        # Submit tasks to the thread pool
        futures = [executor.submit(
            score_matrix,
            matrix, rsat_cmd, seq_file_pos, seq_file_neg, bg_file, tmp_dir) for matrix in matrices]

        # Optionally, wait for all threads to complete and handle exceptions
        for future in concurrent.futures.as_completed(futures):
            future.result()  # This will raise any exceptions that occurred in the threads

        # Collect results into a list
        results = []
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    return results


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
