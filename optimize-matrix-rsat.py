import argparse
import os
import sys
from datetime import datetime

import polars as pl

from src.file_util import check_file
from src.logging import set_verbosity, log_message
from src.matrix_util import parse_transfac
from src.optimize_GA import genetic_algorithm


def main(args):
    # ----------------------------------------------------------------
    # Parameters
    # ----------------------------------------------------------------

    # Set verbosity level
    set_verbosity(args.verbosity)

    # Check the matrix file
    check_file(args.matrices, 'transfac')

    # Check the sequence file
    check_file(args.positives, 'fasta')
    check_file(args.negatives, 'fasta')

    # Create output directory (dirname of output prefix) if it does not exist
    output_dir = os.path.dirname(args.output_prefix)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create directory for temporary files
    tmp_dir = os.path.join(output_dir, "tmp", datetime.now().strftime("%Y-%m-%d_%H%M%S"))
    if not os.path.exists(tmp_dir):
        log_message("info", 2, f"Creating temporary directory {tmp_dir}")
        os.makedirs(tmp_dir)

    # ----------------------------------------------------------------
    # Load original matrices
    # ----------------------------------------------------------------
    log_message("info", 1, f"Loading matrices from file {args.matrices}")
    parsed_matrices = parse_transfac(args.matrices)

    # ----------------------------------------------------------------
    # Run genetic algorithm to optimize the matrices
    # ----------------------------------------------------------------

    if args.selection_level == 'clone':
        matrices = []
        score_table = pl.DataFrame()
        for index, matrix in enumerate(parsed_matrices):
            log_message("info", 0, f'Running genetic algorithm on clone {index + 1}/{len(parsed_matrices)}')
            new_matrices, new_score_table = genetic_algorithm(
                [matrix], args.rsat_cmd, args.positives, args.negatives, args.background,
                output_prefix=f'{args.output_prefix}_{args.selection_level}{index + 1}', tmp_dir=tmp_dir,
                generations=args.generations, select=args.select, children=args.children, threads=args.threads)
            matrices.extend(new_matrices)
            score_table = pl.concat([score_table, new_score_table])
    else:  # args.selection_level == 'population'
        log_message("info", 1, 'Running genetic algorithm on whole population')
        matrices, score_table = genetic_algorithm(
            parsed_matrices, args.rsat_cmd, args.positives, args.negatives, args.background,
            output_prefix=f'{args.output_prefix}_{args.selection_level}', tmp_dir=tmp_dir,
            generations=args.generations, select=args.select, children=args.children, threads=args.threads)

    # Export aggregated score table to TSV
    log_message("info", 1, f"All generations completed")
    matrix_scores_tsv = f"{args.output_prefix}_{args.selection_level}_gen{0}-{args.generations}_score_table.tsv"
    log_message("info", 2, f"Saving score table to {matrix_scores_tsv}")
    score_table.write_csv(matrix_scores_tsv, separator='\t', include_header=True)

    # # Export matrix scores to JSON
    # matrix_scores_json = outfile_prefix + '_matrix_scores.json'
    # log_message("info", 1, f"Saving scored matrices to file {matrix_scores_json}")
    # with open(matrix_scores_json, 'w') as file:
    #     json.dump(final_matrices, file, indent=4)

    # # Export matrix scores to TSV
    # matrix_scores_tsv = outfile_prefix + '_matrix_scores.tsv'
    # matrix_scores_df.write_csv(matrix_scores_tsv, separator='\t')

    log_message("info", 0, f"Job's done.")


if __name__ == '__main__':
    print(*sys.argv)
    description_text = """\
    Optimize position specific scoring matrices according to their capability to discriminate a positive from a negative 
    sequence set. 
    
    Requirement: a local instance of the software suite  Regulatory Sequence Analysis Tools (RSAT), which can be
    installed in various ways (https://rsa-tools.github.io/installing-RSAT/). 
    
    RSAT docker installation:
    
        docker pull eeadcsiccompbio/rsat:20240820 
    
    RSAT docker configuration:

        export RSAT_VERSION=20240820 # use this version or a later one
        export BASE_DIR=$PWD # base directory required for docker to access the data files
        mkdir -p $BASE_DIR
        export RSAT_CMD="docker run -v ${BASE_DIR}:/home/rsat_user -v ${BASE_DIR}/results:/home/rsat_user/out \\
            eeadcsiccompbio/rsat:${RSAT_VERSION} rsat"
        echo ${RSAT_CMD} # check the format
        $RSAT_CMD -h # print rsat help message
    
    Usage example: 
    
        export PYTHON_PATH=venv/bin/python # python path should be adapted to your local settings
        $PYTHON_PATH optimize-matrix-rsat.py -v 3 -t 10 -g 5 -c 5 -s 5 \\
          -m data/matrices/GABPA_CHS_THC_0866_peakmo-clust-trimmed.tf \\
          -p data/sequences/THC_0866.fasta -n data/sequences/THC_0866_rand-loci_noN.fa \\
          -b data/bg_models/equiprobable_1str.tsv -r "${RSAT_CMD}"
    
    """
    parser = argparse.ArgumentParser(description=description_text, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        '-v', '--verbosity', type=int, default=1, help='Verbosity level (int)')
    parser.add_argument(
        '-t', '--threads', type=int, default=1, help='Number of threads (int)')
    parser.add_argument(
        '-g', '--generations', type=int, default=2, help='Number of generations (int)')
    parser.add_argument(
        '-c', '--children', type=int, default=10,
        help='Number of children per generation (int)')
    parser.add_argument(
        '-s', '--select', type=int, default=5,
        help='Number of top matrices to select after each generation (int)')
    parser.add_argument(
        '-m', '--matrices', type=str, required=True,
        help='Transfac-formatted matrix file path (str)')
    parser.add_argument(
        '-p', '--positives', type=str, required=True,
        help='fasta-formatted file containing positive sequences (str)')
    parser.add_argument(
        '-n', '--negatives', type=str, required=True,
        help='fasta-formatted file containing negative sequences (str)')
    parser.add_argument(
        '-b', '--background', type=str, required=True,
        help='rsat oligo-analysis formatted background model file (str)')
    parser.add_argument(
        '-r', '--rsat_cmd', type=str, required=True,
        help='RSAT command, either a full path or a container (docker, Apptainer) with parameters (str)')
    parser.add_argument(
        '--selection_level', type=str, choices=['population', 'clone'], default='population',
        help='Selection level. Accepted values: population (default) or clone')
    parser.add_argument(
        '-o', '--output_prefix', type=str, required=True,
        help='Output prefix. It can contain a folder path, which is created if not existing (str)')

    main(parser.parse_args())
