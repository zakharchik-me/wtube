import yaml
import os


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def get_video_filename(config):
    video_path = config.get('source', {}).get('args', {}).get('path', '')
    return os.path.basename(video_path)