# STAGE 2: Multi-Source Data Acquisition
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")

# List of 7 project IDs (IR, UV, and Standard spectrums)
project_ids = ["soil-moisture-ir", "soil-moisture-uv", "standard-laser-data"] # simplified list
for pid in project_ids:
    project = rf.workspace("your-workspace").project(pid)
    dataset = project.version(1).download("yolov5")
