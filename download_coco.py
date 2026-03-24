import yaml
from pathlib import Path
from ultralytics.utils.downloads import download

yaml_path = Path('ultralytics/cfg/datasets/coco.yaml')
with open(yaml_path) as f:
    yaml_data = yaml.safe_load(f)

print("Starting COCO download...")
# The download script in coco.yaml expects a 'yaml' dict with 'path'
# We'll execute the script string from the yaml file.
# Note: The script uses 'yaml' variable which we provide in globals.
exec(yaml_data['download'], {'yaml': {'path': str(Path('datasets/coco'))}})
