import os
import sys

from src.fasta_utils import validate_fasta
from src.logging import log_message
from src.matrix_util import validate_transfac


def check_file(file_path, expected_format):
    """
    Check if the file exists, is not empty, is a text file, and matches the expected format.

    Parameters:
    -----------
    file_path : str
        The path to the file that needs to be checked.
    expected_format : str
        The expected format of the file. Should be either 'transfac' or 'fasta'.

    Returns:
    --------
    None
        This function does not return any value. It exits the program with an error message if any check fails.

    Notes:
    ------
    This is ChatGPT-generated code.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        sys.exit(1)

    # Check if file is not empty
    if os.path.getsize(file_path) == 0:
        print(f"Error: The file '{file_path}' is empty.")
        sys.exit(1)

    # Check if file is a text file
    try:
        with open(file_path, 'r') as f:
            f.read(1024)  # Try reading the first 1KB to ensure it's a text file
    except UnicodeDecodeError:
        print(f"Error: The file '{file_path}' is not a text file.")
        sys.exit(1)

    # Validate format based on the expected format
    if expected_format == 'transfac':
        if not validate_transfac(file_path):
            print(f"Error: The file '{file_path}' is not in the correct TRANSFAC format.")
            sys.exit(1)
    elif expected_format == 'fasta':
        if not validate_fasta(file_path):
            print(f"Error: The file '{file_path}' is not in the correct FASTA format.")
            sys.exit(1)
    else:
        print(f"Error: Unsupported format '{expected_format}'.")
        sys.exit(1)

    log_message("info", 2,
                f"File '{file_path}' passed all checks ({expected_format} format)")
