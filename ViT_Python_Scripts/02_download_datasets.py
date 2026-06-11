"""
Script 02: Download all seven Roboflow datasets
Soil Moisture Classification — ViT + YOLOv8 Pipeline
"""

import os
from roboflow import Roboflow

BASE_DIR  = '/kaggle/working/source_data'
os.makedirs(BASE_DIR, exist_ok=True)

rf = Roboflow(api_key="yRqyBbimhh1vgoeZs2Gx")

projects = [
    ("robotics-lab-1", "soil-moisture-v4",              3),
    ("robotics-lab-1", "soil-moisture-v4-ir",           1),
    ("robotics-lab-1", "soil-moisture-v4-uv",           1),
    ("robotics-lab-1", "soil-moisture-ir",              1),
    ("robotics-lab-1", "soil-moisture-5sagf",           1),
    ("robotics-lab-1", "soil_moisture_september",       8),
    ("robotics-lab-1", "soil_moisture_stir_september",  4),
]

for workspace, proj_name, ver in projects:
    try:
        project = rf.workspace(workspace).project(proj_name)
        dataset = project.version(ver).download(
            "yolov5",
            location=os.path.join(BASE_DIR, proj_name)
        )
        print(f"Downloaded: {proj_name} v{ver}")
    except Exception as e:
        print(f"Skipping {proj_name}: {e}")

print("\nAll downloads complete.")
