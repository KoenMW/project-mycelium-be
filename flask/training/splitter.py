import os, re, shutil
from typing import List

def split_by_hour(source_folder: str, output_folder: str, hour_split: int = 24):
    """
    Organize training images into class folders based on time progression.
    
    This function converts a flat directory of timestamped images into a supervised
    learning dataset by grouping images into time-based classes. Each class represents
    a growth stage (e.g., day 0, day 1, day 2, etc.) determined by elapsed time.
    
    Args:
        source_folder (str): Directory containing raw training images with standardized naming
        output_folder (str): Directory to create class-based folder structure
        hour_split (int): Hours per class bucket (default: 24 = daily growth stages)
        
    Returns:
        None: Creates organized directory structure on disk
    """
    # === Clean Setup ===
    # Remove existing output directory to start fresh
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # === File Discovery and Filtering ===
    # Find all image files that match the expected naming convention
    # Expected format: test{num}_h{hour}_{angle}.jpg
    files = [f for f in os.listdir(source_folder) if re.match(r'test\d+_h\d+_\d+\.jpg', f)]
    file_hour_map = {}  # Dictionary to group files by test number

    # === Parse File Metadata ===
    # Extract test number and hour information from each filename
    for f in files:
        # Use regex to parse standardized filename format
        match = re.match(r'test(\d+)_h(\d+)_(\d+)\.jpg', f)
        if not match:
            continue  # Skip files that don't match naming convention
            
        # Extract metadata components
        test_num = int(match.group(1))  # Experiment/test identifier
        hour = int(match.group(2))      # Time when image was captured
        angle = int(match.group(3))     # Camera angle (not used for splitting)
        
        # Group files by test number for per-experiment processing
        file_hour_map.setdefault(test_num, []).append((f, hour))

    # === Time-Based Class Assignment ===
    # Process each test/experiment separately to handle different start times
    for test, file_list in file_hour_map.items():
        if not file_list:
            continue  # Skip empty test groups
            
        # === Calculate Relative Time ===
        # Find the earliest timestamp for this experiment as the reference point
        hours = [h for _, h in file_list]
        min_hour = min(hours)  # Experiment start time (hour 0)
        
        # === Assign Images to Class Buckets ===
        # Organize images into growth stage classes based on elapsed time
        for f, hour in file_list:
            # Calculate elapsed time from experiment start
            elapsed_hours = hour - min_hour
            
            # Determine which class bucket this image belongs to
            # bucket 0 = hours 0-23 (day 0), bucket 1 = hours 24-47 (day 1), etc.
            bucket = elapsed_hours // hour_split
            
            # === Create Class Directory and Copy Image ===
            # Create class folder if it doesn't exist (named by bucket number)
            dst_dir = os.path.join(output_folder, str(bucket))
            os.makedirs(dst_dir, exist_ok=True)
            
            # Copy image to appropriate class folder
            src_path = os.path.join(source_folder, f)
            dst_path = os.path.join(dst_dir, f)
            shutil.copy2(src_path, dst_path)  # copy2 preserves metadata