import sys

from loguru import logger

# ----------------------------------------------------------------
# Handle logging
# ----------------------------------------------------------------
# Remove all existing handlers to avoid duplication
logger.remove()

# Add a single handler with DEBUG level to handle all message types
logger.add(sys.stdout, format="{time} {level} {message}", level="DEBUG")

# Global verbosity level
VERBOSITY = 1


def set_verbosity(level):
    global VERBOSITY
    VERBOSITY = level


def get_indentation(level):
    """Return the number of spaces for indentation based on verbosity level."""
    return " " * (level * 4)  # 4 spaces per level


def log_message(msg_type, level, message):
    """Log a message with indentation based on verbosity level."""
    if VERBOSITY >= level:
        indentation = get_indentation(level)
        formatted_message = f"{indentation}{message}"
        # Map msg_type to the corresponding Loguru method
        if msg_type == "info":
            logger.info(formatted_message)
        elif msg_type == "warning":
            logger.warning(formatted_message)
        elif msg_type == "debug":
            logger.debug(formatted_message)
        else:
            print(formatted_message)
