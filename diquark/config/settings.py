import yaml
from pathlib import Path

def load_settings(settings_path: str = "config/default_settings.yaml") -> dict:
    base_dir = Path(__file__).resolve().parent.parent
    with open(base_dir / settings_path, 'r') as f:
        settings = yaml.safe_load(f)
    
    # Set the base directory
    settings['data']['base_dir'] = str(base_dir)
    
    # Convert relevant paths to Path objects
    settings['data']['data_dir'] = base_dir / settings['data']['data_dir']
    settings['data']['output_dir'] = base_dir / settings['data']['output_dir']
    
    return settings

# Load the settings
DEFAULT_SETTINGS = load_settings()