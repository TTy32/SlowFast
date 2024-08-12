import os
import pandas as pd

# Global list to store unique strings
unique_strings_global = []

def extract_unique_strings(file_path, filenames):
  
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path, header=None)
    
    # Extract the first column
    first_column = df[0]
    
    # Get unique strings from the first column
    unique_strings = first_column.unique()
    
    # Add unique strings to the global list if not already present
    for string in unique_strings:
        if string not in filenames:
            filenames.append(string)

def append_filenames_from_file(file_path, filenames):
    # Read the file line by line
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Strip newlines and add unique filenames to the global list
    for line in lines:
        filename = line.strip()
        if filename not in filenames:
            filenames.append(filename)


def left_join_with_extensions(filenames, extensions_file_path):
    # Create a dictionary to map filenames without extensions to their full versions
    filename_extension_map = {}
    
    # Read the file with extensions line by line
    with open(extensions_file_path, 'r') as file:
        lines = file.readlines()
    
    # Populate the dictionary with the filename without extension as the key
    for line in lines:
        full_filename = line.strip()
        name_without_extension = os.path.splitext(full_filename)[0]  # Extract the name without the extension
        filename_extension_map[name_without_extension] = full_filename
    
    # Update the filenames list with extensions if available
    for i, filename in enumerate(filenames):
        if filename in filename_extension_map:
            filenames[i] = filename_extension_map[filename]

filenames = []
base_dir = '../ava/annotations/'
extract_unique_strings(os.path.join(base_dir, 'ava_train_v2.2.csv'), filenames)
extract_unique_strings(os.path.join(base_dir, 'ava_val_v2.2.csv'), filenames)
append_filenames_from_file(os.path.join(base_dir, 'ava_test_v2.2.csv'), filenames)

# Last
left_join_with_extensions(filenames, os.path.join('../ava', 'ava_file_names_trainval_v2.1.txt'))


print (filenames)
print("Filenames: " + str(len(filenames)))



