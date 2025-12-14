import logging
import os
import sys

# Directory for all outputs (logs, plots, models)
OUTPUT_DIR = "output"

def setup_logging(log_file_name: str = "run.log"):
    """
    Configures logging to write to a file in OUTPUT_DIR and to the console.
    
    Args:
        log_file_name (str): The name of the log file to create (e.g., "data.log", "train.log")
    """
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) # Set the minimum logging level
    
    # Clear existing handlers (if any, useful for re-running in notebooks)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_file_path = os.path.join(OUTPUT_DIR, log_file_name)

    # Create file handler (writes to file)
    # mode='w' will overwrite the log file each time
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Create console handler (writes to terminal)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s')) # Simple format for console
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logging.info(f"Logging configured. Output will be saved to {log_file_path}")
