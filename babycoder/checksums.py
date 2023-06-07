import hashlib
import os
import json

import hashlib
import os
import json


def calculate_checksums(start_path):
    """
    Recursively calculates checksums for all files in a given folder and its subfolders.
    
    Args:
        start_path (str): The path to the folder where the calculation should begin.
    
    Returns:
        dict: A dictionary mapping file paths to their corresponding checksums.
    """
    checksums = {}
    
    for root, _, files in os.walk(start_path):
        for file in files:
            file_path = os.path.join(root, file)
            
            with open(file_path, 'rb') as f:
                contents = f.read()
                checksum = hashlib.sha256(contents).hexdigest()
                
            checksums[file_path] = checksum
    
    return checksums


def compare_checksums(start_path, stored_checksums):
    """
    Compares the checksums of files in a given folder and its subfolders to the stored checksums.
    
    Args:
        start_path (str): The path to the folder where the comparison should begin.
        stored_checksums (dict): A dictionary mapping file paths to their corresponding stored checksums.
    
    Returns:
        dict: A dictionary with three lists:
            - 'changed': Contains file paths that have a different checksum compared to the stored checksum
                         or files that no longer exist.
            - 'unchanged': Contains file paths that have the same checksum as the stored checksum.
            - 'deleted': Contains file paths that existed in the stored checksums but are no longer present.
    """
    changed_files = []
    unchanged_files = []    
    for root, _, files in os.walk(start_path):
        for file in files:
            file_path = os.path.join(root, file)
            
            with open(file_path, 'rb') as f:
                contents = f.read()
                checksum = hashlib.sha256(contents).hexdigest()
                
            stored_checksum = stored_checksums.get(file_path)
            
            if stored_checksum is None or checksum != stored_checksum:
                changed_files.append(file_path)
            else:
                unchanged_files.append(file_path)
    
    for stored_file_path in stored_checksums:
        if not os.path.exists(stored_file_path):
            changed_files.append(stored_file_path)
    
    return {'changed': changed_files, 'unchanged': unchanged_files}


def persist_checksums(checksums, file_path):
    """
    Persists the checksums dictionary to a JSON file.
    
    Args:
        checksums (dict): A dictionary mapping file paths to their corresponding checksums.
        file_path (str): The path to the file where the checksums should be saved.
    """
    with open(file_path, 'w') as f:
        json.dump(checksums, f)



def load_checksums(file_path):
    """
    Loads the checksums dictionary from a JSON file.

    Args:
        file_path (str): The path to the file containing the stored checksums.

    Returns:
        dict: A dictionary mapping file paths to their corresponding checksums, or an empty dictionary if the file
              does not exist.
    """
    if not os.path.exists(file_path):
        return {}

    with open(file_path, 'r') as f:
        checksums = json.load(f)

    return checksums
