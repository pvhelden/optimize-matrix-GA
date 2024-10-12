import io
import subprocess

import polars as pl

from src.logging import log_message


def run_command(command):
    log_message("info", 4, f"Running command\n\t{format(command)}")
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    return result


def scan_sequences_rsat(rsat_cmd, seq_file, label, matrix_file, bg_file,
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
    log_message("info", 4, f"Running scan command with matrix file {matrix_file} and label {label}")
    # Run the command
    scan_result = run_command(scan_cmd)

    # Catch the stout in a buffer to enable post-processing with pl.read_csv
    # - filter out comment lines
    # - get the header
    # - select only required columns
    csv_buffer = io.StringIO(scan_result.stdout)
    result = pl.read_csv(
        csv_buffer,
        comment_prefix=';',
        has_header=True,
        separator='\t',
    ).select([
        pl.col(score_col),
        pl.col(group_col),
        pl.lit(label).alias('label'),
    ]).filter(pl.col(score_col) >= score_threshold)

    return result
