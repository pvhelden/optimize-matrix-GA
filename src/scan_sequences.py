import numpy as np
import polars as pl
from tqdm import tqdm

from src.fasta_utils import read_fasta
from src.matrix_util import parse_transfac, pcm_to_pwm


def scan_sequences(seq_file, label, matrix_file, bg_file, score_col='weight', group_col='ft_name',
                   score_threshold=-100):
    matrices = parse_transfac(matrix_file)
    for matrix_dic in matrices:
        matrix_dic['pwm'] = pcm_to_pwm(matrix_dic['matrix'])

    results = []
    with tqdm() as pbar:
        for header, seq in read_fasta(seq_file):
            pbar.update()
            for matrix_dic in matrices:
                best_score = -np.inf
                pwm = matrix_dic['pwm']
                motif_size = pwm.shape[1]
                for pos in range(0, len(seq) - motif_size + 1):
                    substring = seq[pos:pos + motif_size]
                    score = 0
                    for index, char in enumerate(substring):
                        score += pwm[index, char.capitalize()]
                    if score > best_score:
                        best_score = score

                if score >= score_threshold:
                    results.append({score_col: best_score, group_col: matrix_dic['metadata']['ID'], 'label': label})

    return pl.DataFrame(results)
