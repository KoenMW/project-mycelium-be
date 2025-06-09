import os, re, shutil
from typing import List

def split_by_hour(source_folder: str, output_folder: str, hour_split: int = 24):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    files = [f for f in os.listdir(source_folder) if re.match(r'test\d+_h\d+_\d+\.jpg', f)]
    file_hour_map = {}

    for f in files:
        match = re.match(r'test(\d+)_h(\d+)_(\d+)\.jpg', f)
        if not match:
            continue
        test_num = int(match.group(1))
        hour = int(match.group(2))
        file_hour_map.setdefault(test_num, []).append((f, hour))

    for test, file_list in file_hour_map.items():
        if not file_list:
            continue
        hours = [h for _, h in file_list]
        min_hour = min(hours)
        for f, hour in file_list:
            bucket = (hour - min_hour) // hour_split
            dst = os.path.join(output_folder, str(bucket))
            os.makedirs(dst, exist_ok=True)
            shutil.copy2(os.path.join(source_folder, f), os.path.join(dst, f))
