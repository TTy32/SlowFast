import os
import shutil
import csv
import json

# Specify variables
OUTPUT_DIR = "somethingsomething-trunc"
NUMBER_OF_VIDEOS = 256
NUMBER_OF_FRAMES_PER_VIDEO = 180
MOCK_JPEG_PATH = "./mock.jpg"

def generate_video_data(output_dir, num_videos, num_frames_per_video, mock_jpeg_path, csv_filename, json_filename, starting_video_id=1):
    # Create necessary directories
    os.makedirs(f"{output_dir}/frames", exist_ok=True)

    # Initialize the CSV and JSON data structures
    csv_rows = []
    json_data = []

    # Generate the data
    for video_id_num in range(starting_video_id, starting_video_id + num_videos):
        video_id = f"{video_id_num:05d}"
        video_frames_dir = f"{output_dir}/frames/{video_id}"
        os.makedirs(video_frames_dir, exist_ok=True)
        
        for frame_id_num in range(num_frames_per_video):
            frame_id_str = f"{frame_id_num:06d}"
            frame_filename = f"{video_id}_{frame_id_str}.jpg"
            frame_path = f"{video_frames_dir}/{frame_filename}"
            frame_path_csv = f"{video_id}/{frame_filename}"
            
            # Copy the mock image to the frame path
            shutil.copyfile(mock_jpeg_path, frame_path)
            
            # Add row to CSV
            csv_rows.append([video_id, "222222", frame_id_num, frame_path_csv, '"'])
        
        # Add entry to JSON
        json_data.append({
            "id": video_id,
            "label": "holding bulb",
            "template": "Holding [something]",
            "placeholders": ["bulb"]
        })

    # Write to the CSV file with space-separated values
    csv_path = f"{output_dir}/{csv_filename}"
    with open(csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=' ', doublequote=False, escapechar='"')
        csv_writer.writerow(["original_vido_id", "video_id", "frame_id", "path", "labels"]) # YES, its 'original_vido_id' (include the typo)
        csv_writer.writerows(csv_rows)

    # Write to the JSON file
    json_path = f"{output_dir}/{json_filename}"
    with open(json_path, mode='w') as json_file:
        json.dump(json_data, json_file, indent=4)

generate_video_data(
    output_dir=OUTPUT_DIR,
    num_videos=NUMBER_OF_VIDEOS,
    num_frames_per_video=NUMBER_OF_FRAMES_PER_VIDEO,
    mock_jpeg_path=MOCK_JPEG_PATH,
    csv_filename="train.csv",
    json_filename="something-something-v2-train.json",
    starting_video_id=1
)

generate_video_data(
    output_dir=OUTPUT_DIR,
    num_videos=NUMBER_OF_VIDEOS,
    num_frames_per_video=NUMBER_OF_FRAMES_PER_VIDEO,
    mock_jpeg_path=MOCK_JPEG_PATH,
    csv_filename="val.csv",
    json_filename="something-something-v2-validation.json",
    starting_video_id=NUMBER_OF_VIDEOS +1
)
