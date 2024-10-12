def read_fasta(file_path):
    """
    Generator function to read sequences from a FASTA file one by one.

    Parameters:
    file_path (str): Path to the FASTA file.

    Yields:
    tuple: A tuple containing the sequence header and the sequence.
    """
    header = None
    sequence = []

    with open(file_path, 'r') as fasta_file:
        for line in fasta_file:
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            if line.startswith(">"):
                # If this is not the first header, yield the previous sequence
                if header:
                    yield header, ''.join(sequence)
                header = line[1:]  # Capture the header, removing the ">"
                sequence = []  # Reset sequence for new entry
            else:
                sequence.append(line)  # Add sequence lines to the list

        # Yield the last sequence in the file
        if header:
            yield header, ''.join(sequence)


def validate_fasta(file_path):
    """
    Validate if the file is in FASTA format.

    Parameters:
    -----------
    file_path : str
        The path to the file that needs to be validated.

    Returns:
    --------
    bool
        Returns True if the file is in the correct FASTA format, otherwise False.

    Notes:
    ------
    This is a basic check and might need to be extended for full compliance.
    This is ChatGPT-generated code.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        # FASTA typically starts with a '>' character
        if not lines or not lines[0].startswith('>'):
            return False
    return True
