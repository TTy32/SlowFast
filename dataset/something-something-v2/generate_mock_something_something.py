import os
import shutil
import csv
import json

# Specify variables
OUTPUT_DIR = "somethingsomething-trunc2"
NUMBER_OF_VIDEOS = 2
NUMBER_OF_FRAMES_PER_VIDEO = 4
MOCK_JPEG_PATH = "./mock.jpg"

# Create necessary directories
os.makedirs(f"{OUTPUT_DIR}/frames", exist_ok=True)

# Initialize the CSV and JSON data structures
csv_rows = []
json_data = []

# Generate the data
for video_id_num in range(1, NUMBER_OF_VIDEOS + 1):
    video_id = f"{video_id_num:05d}"
    video_frames_dir = f"{OUTPUT_DIR}/frames/{video_id}"
    os.makedirs(video_frames_dir, exist_ok=True)
    
    for frame_id_num in range(NUMBER_OF_FRAMES_PER_VIDEO):
        frame_id_str = f"{frame_id_num:06d}"
        frame_filename = f"{video_id}_{frame_id_str}.jpg"
        frame_path = f"{video_frames_dir}/{frame_filename}"
        frame_path_csv = f"{video_id}/{frame_filename}"
        
        # Copy the mock image to the frame path
        shutil.copyfile(MOCK_JPEG_PATH, frame_path)
        
        # Add row to CSV
        csv_rows.append([video_id, "222222", frame_id_num, frame_path_csv, ""])
    
    # Add entry to JSON
    json_data.append({
        "id": video_id,
        "label": "holding bulb",
        "template": "Holding [something]",
        "placeholders": ["bulb"]
    })

# Write to train.csv with space-separated values
csv_path = f"{OUTPUT_DIR}/train.csv"
with open(csv_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=' ')
    csv_writer.writerow(["original_video_id", "video_id", "frame_id", "path", "labels"])
    csv_writer.writerows(csv_rows)

# Write to something-something-v2-train.json
json_path = f"{OUTPUT_DIR}/something-something-v2-train.json"
with open(json_path, mode='w') as json_file:
    json.dump(json_data, json_file, indent=4)
