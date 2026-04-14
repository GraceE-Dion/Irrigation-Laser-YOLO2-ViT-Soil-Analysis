# STEP 2: Multi-Source Data Acquisition
rf = Roboflow(api_key="yRqyBbimhh1vgoeZs2Gx")

projects = [
    ("robotics-lab-1", "soil-moisture-v4", 3),
    ("robotics-lab-1", "soil-moisture-v4-ir", 1),
    ("robotics-lab-1", "soil-moisture-v4-uv", 1),
    ("robotics-lab-1", "soil-moisture-ir", 1),
    ("robotics-lab-1", "soil-moisture-5sagf", 1),
    ("robotics-lab-1", "soil_moisture_september", 4),
    ("robotics-lab-1", "soil_moisture_stir_september", 1)
]

BASE_DIR = '/kaggle/working/source_data'
MASTER_DIR = '/kaggle/working/Master_Soil_Moisture'
os.makedirs(BASE_DIR, exist_ok=True)

for workspace, proj_name, ver in projects:
    try:
        project = rf.workspace(workspace).project(proj_name)
        dataset = project.version(ver).download("yolov5", 
                  location=os.path.join(BASE_DIR, proj_name))
    except Exception as e:
        print(f"Skipping {proj_name}: {e}")
