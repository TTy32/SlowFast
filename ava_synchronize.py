import os

# Base directory
base_dir = os.path.expanduser('~/ava2')

# Path to the directory containing the frame directories
frames_dir = os.path.join(base_dir, 'frames')

# List of all frame directory names
frame_dirs = set(os.listdir(frames_dir))

# List of files to process
files_to_process = [
    'frame_lists/train.csv',
    'annotations/ava_train_v2.1.csv',
    'annotations/person_box_67091280_iou90/ava_train_v2.1.csv',
    'annotations/person_box_67091280_iou90/ava_detection_train_boxes_and_labels_include_negative_v2.2.csv',
    'annotations/person_box_67091280_iou90/train.csv',
    'annotations/person_box_67091280_iou90/ava_detection_train_boxes_and_labels_include_negative_v2.1.csv',
    'annotations/person_box_67091280_iou90/ava_train_predicted_boxes.csv',
    'annotations/person_box_67091280_iou90/ava_detection_train_boxes_and_labels_include_negative.csv',
    'annotations/ava_train_v2.2.csv',
    'annotations/person_box_67091280_iou75/ava_detection_train_boxes_and_labels_include_negative_v2.2.csv',
    'annotations/person_box_67091280_iou75/ava_detection_train_boxes_and_labels_include_negative_v2.1.csv',
    'annotations/person_box_67091280_iou75/ava_detection_train_boxes_and_labels_include_negative.csv',
    'annotations/train.csv',
    'annotations/ava_train_predicted_boxes.csv'
]

def filter_lines(file_path, frame_dirs):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    filtered_lines = [line for line in lines if any(frame_dir in line for frame_dir in frame_dirs)]
    
    with open(file_path, 'w') as file:
        file.writelines(filtered_lines)

# Process each file
for relative_path in files_to_process:
    absolute_path = os.path.join(base_dir, relative_path)
    filter_lines(absolute_path, frame_dirs)

print("Processing complete. Files have been updated.")
