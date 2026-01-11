import os
import json
import shutil
import logging
from datetime import datetime
import glob

logger = logging.getLogger(__name__)

class ModelRegistry:
    def __init__(self, registry_path="models/registry"):
        self.registry_path = registry_path
        self.pointer_file = os.path.join(registry_path, "current.json")
        os.makedirs(registry_path, exist_ok=True)
        
    def create_version(self, artifacts, metrics, config_dump=None):
        """
        Create a new version folder and save artifacts.
        artifacts: dict of {filename: source_path} or {filename: content}
        metrics: dict
        """
        # 1. Generate Version ID
        version_id = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        version_dir = os.path.join(self.registry_path, version_id)
        os.makedirs(version_dir, exist_ok=True)
        
        logger.info(f"Creating model version: {version_id}")
        
        # 2. Copy/Save Artifacts
        for filename, source in artifacts.items():
            dest = os.path.join(version_dir, filename)
            if os.path.exists(source):
                # source is a file path
                shutil.copy(source, dest)
            else:
                # source is content (e.g. string or dict)
                with open(dest, "w") as f:
                    if isinstance(source, dict) or isinstance(source, list):
                        json.dump(source, f, indent=4)
                    else:
                        f.write(str(source))
                        
        # 3. Save Metrics
        with open(os.path.join(version_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)
            
        # 4. Save Config
        if config_dump:
            with open(os.path.join(version_dir, "config.yaml"), "w") as f:
                if isinstance(config_dump, dict):
                    json.dump(config_dump, f, indent=4) # Fallback to json if dict passed
                else:
                    f.write(config_dump)
                    
        # 5. Metadata
        with open(os.path.join(version_dir, "created_at.txt"), "w") as f:
            f.write(datetime.now().isoformat())
            
        return version_id
        
    def promote_version(self, version_id):
        """
        Point 'current.json' to this version.
        """
        version_dir = os.path.join(self.registry_path, version_id)
        if not os.path.exists(version_dir):
            raise ValueError(f"Version {version_id} does not exist.")
            
        current_data = {
            "current_version": version_id,
            "updated_at": datetime.now().isoformat(),
            "path": os.path.abspath(version_dir)
        }
        
        # Load history
        history = []
        if os.path.exists(self.pointer_file):
            try:
                with open(self.pointer_file, "r") as f:
                    old = json.load(f)
                    history = old.get("history", [])
                    # Append previous current to history
                    if "current_version" in old:
                         history.append({
                             "version": old["current_version"],
                             "promoted_at": old["updated_at"]
                         })
            except: pass
            
        current_data["history"] = history
        
        with open(self.pointer_file, "w") as f:
            json.dump(current_data, f, indent=4)
            
        logger.info(f"Promoted {version_id} to Production.")
        
    def rollback(self):
        """
        Revert to the previous version in history.
        """
        if not os.path.exists(self.pointer_file):
            logger.warning("No current version to rollback from.")
            return False
            
        with open(self.pointer_file, "r") as f:
            data = json.load(f)
            
        history = data.get("history", [])
        if not history:
            logger.warning("No history available for rollback.")
            return False
            
        # Pop last item
        last_version = history.pop()
        target_version = last_version["version"]
        
        logger.info(f"Rolling back to version: {target_version}")
        
        # Check existence
        version_dir = os.path.join(self.registry_path, target_version)
        if not os.path.exists(version_dir):
            logger.error(f"Target rollback version {target_version} not found on disk.")
            return False
            
        # Update pointer
        data["current_version"] = target_version
        data["updated_at"] = datetime.now().isoformat()
        data["path"] = os.path.abspath(version_dir)
        data["history"] = history # Update history (removed last)
        
        with open(self.pointer_file, "w") as f:
            json.dump(data, f, indent=4)
            
        return True

    def get_current_model_path(self):
        if not os.path.exists(self.pointer_file): return None
        with open(self.pointer_file, "r") as f:
            data = json.load(f)
        return data.get("path")
