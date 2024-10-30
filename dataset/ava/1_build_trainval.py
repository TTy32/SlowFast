import os
import pandas as pd
import requests
import urllib.parse

def check_and_add_extension(filenames, base_url='https://s3.amazonaws.com/ava-dataset/trainval/'):
    # Test extensions
    extensions = ['mp4', 'mkv', 'webm']
    
    # Iterate over filenames without extensions
    for i, filename in enumerate(filenames):
        if '.' not in filename:  # If the filename has no extensio
            for ext in extensions:
                encoded_filename = urllib.parse.quote(f"{filename}.{ext}")
                url = f"{base_url}{encoded_filename}"
                try:
                    response = requests.head(url, timeout=1)
                    #print (url)
                    if response.status_code == 200:  # URL exists
                        print ('FOUND ' + filename)
                        filenames[i] = f"{filename}.{ext}"
                        break
                except requests.RequestException as e:
                    print(f"Request to {url} failed: {e}")


def extract_unique_strings(file_path, filenames):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path, header=None)

    # Extract the first column
    first_column = df[0]
    
    # Get the base filenames (without extensions) currently in the list
    base_filenames = {os.path.splitext(f)[0] for f in filenames}
    
    # Get unique strings from the first column
    unique_strings = first_column.unique()
    
    # Add unique strings to the list if not already present (match only base filename)
    for string in unique_strings:
        if string not in base_filenames:
            filenames.append(string)

def append_filenames_from_file(file_path, filenames):
    # Get the base filenames (without extensions) currently in the list
    base_filenames = {os.path.splitext(f)[0] for f in filenames}
    
    # Read the file line by line
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Strip newlines and add unique filenames to the list if not already present
    for line in lines:
        filename = line.strip()
        if filename not in base_filenames:
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

def read_filenames_txt(file_path, filenames):
    # Read the existing filenames.txt if it exists
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Add each line to the filenames list
        for line in lines:
            filename = line.strip()
            if filename not in filenames:
                filenames.append(filename)

def write_filenames_txt(file_path, filenames):
    # Write the filenames to filenames.txt
    with open(file_path, 'w') as file:
        for filename in filenames:
            file.write(f"{filename}\n")

# Initialize the list to store unique filenames
filenames = []
base_dir = '../ava/annotations/'

# Read from existing filenames.txt if available
read_filenames_txt(os.path.join(base_dir, 'filenames.txt'), filenames)

# Extract unique strings from CSV files
extract_unique_strings(os.path.join(base_dir, 'ava_train_v2.2.csv'), filenames)
extract_unique_strings(os.path.join(base_dir, 'ava_val_v2.2.csv'), filenames)

# Append additional filenames from a file
append_filenames_from_file(os.path.join(base_dir, 'ava_test_v2.2.csv'), filenames)

# Perform the left join with the file that contains extensions
left_join_with_extensions(filenames, os.path.join('../ava', 'ava_file_names_trainval_v2.1.txt'))

# Check and add extensions by validating the URL
check_and_add_extension(filenames)

# Write the updated list of filenames to filenames.txt
write_filenames_txt(os.path.join(base_dir, 'filenames.txt'), filenames)

# Print the final list of filenames and their count
print(filenames)
print("Filenames: " + str(len(filenames)))



