
import os
import fnmatch

def find_files_with_patterns(directory, patterns, n_files=2):
    ''' return a list containg directories of `n_files` images from each lead format from the test dataset.'''

    matched_files = {pattern: [] for pattern in patterns}

    for file in os.listdir(directory):
        # Check if we've already found two files for this pattern
        for pattern in patterns:
            if len(matched_files[pattern]) >= n_files:
                continue  # Move to the next pattern if already found 2 files
            
            # Check if the file matches the current pattern
            if fnmatch.fnmatch(file, f"*{pattern}"):
                matched_files[pattern].append(directory+'/'+file)
    
    image_paths_for_inference = []
    for pattern, files in matched_files.items():
        image_paths_for_inference.extend(files)

    return image_paths_for_inference


