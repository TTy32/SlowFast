import os
import csv
import shutil


base_dir = "."

try:
    frame_dirs = set(os.listdir( os.path.join(base_dir, 'frames') ))
except:
    frame_dirs = set(os.listdir( os.path.join(base_dir, 'rawframes') ))

files_to_process = [
    'frame_lists/train.csv',
    'frame_lists/val.csv',
    'annotations/ava_train_v2.1.csv',
    'annotations/person_box_67091280_iou90/ava_train_v2.1.csv',
    'annotations/person_box_67091280_iou90/ava_detection_train_boxes_and_labels_include_negative_v2.2.csv',
    'annotations/person_box_67091280_iou90/train.csv',
    'annotations/person_box_67091280_iou90/ava_detection_train_boxes_and_labels_include_negative_v2.1.csv',
    'annotations/person_box_67091280_iou90/ava_train_predicted_boxes.csv',
    'annotations/person_box_67091280_iou90/ava_detection_train_boxes_and_labels_include_negative.csv',
    'annotations/person_box_67091280_iou90/ava_detection_val_boxes_and_labels.csv',
    'annotations/ava_train_v2.2.csv',
    'annotations/person_box_67091280_iou75/ava_detection_train_boxes_and_labels_include_negative_v2.2.csv',
    'annotations/person_box_67091280_iou75/ava_detection_train_boxes_and_labels_include_negative_v2.1.csv',
    'annotations/person_box_67091280_iou75/ava_detection_train_boxes_and_labels_include_negative.csv',
    'annotations/train.csv',
    'annotations/ava_train_predicted_boxes.csv',
    'annotations/ava_file_names_trainval_v2.1.txt',
    'annotations/ava_file_names_trainval_v2.1.txt.1',
    'annotations/ava_file_names_trainval_v2.1.txt.2',
    'annotations/ava_included_timestamps_v2.2.txt',
    'annotations/ava_test_excluded_timestamps_v2.2.csv',
    'annotations/ava_test_v2.2.txt',
    'annotations/ava_train_excluded_timestamps_v2.2.csv',
    'annotations/ava_train_v2.2.csv',
    'annotations/ava_val_excluded_timestamps_v2.2.csv',
    'annotations/ava_val_v2.2.csv'
]

def detect_delimiter_and_header(file_path):
    with open(file_path, 'r') as file:
        # Try reading the first line as space-separated
        first_line = file.readline()
        if ' ' in first_line and ',' not in first_line:
            delimiter = ' '
        else:
            delimiter = ','

        # Rewind to start of the file
        file.seek(0)
        
        # Determine if there's a header
        reader = csv.reader(file, delimiter=delimiter)
        first_row = next(reader)
        header_detected = any(col.isalpha() for col in first_row)
        
    return delimiter, header_detected

def filter_lines(file_path, frame_dirs):
    delimiter, header_detected = detect_delimiter_and_header(file_path)
    temp_file_path = file_path + '.tmp'
    
    with open(file_path, 'r') as infile, open(temp_file_path, 'w', newline='') as outfile:
        reader = csv.reader(infile, delimiter=delimiter)
        writer = csv.writer(outfile, delimiter=delimiter)

        # If there's a header, write it first
        if header_detected:
            header = next(reader)
            writer.writerow(header)
        
        for row in reader:
            # Check if any frame directory is present in the relevant column
            if any(frame_dir in row[0] for frame_dir in frame_dirs):
                writer.writerow(row)
    
    # Replace the original file with the filtered file
    shutil.move(temp_file_path, file_path)

# Process each file
for relative_path in files_to_process:
    print(f">> Processing {relative_path}")
    absolute_path = os.path.join(base_dir, relative_path)
    if os.path.isfile(absolute_path):
        filter_lines(absolute_path, frame_dirs)

print("Processing complete. Files have been updated.")