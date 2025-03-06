import os
import json
from logging import getLogger

logger = getLogger(__name__)


class CVDatabase:
    def __init__(self, storage_dir):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

    def save_cv(self, cv_id, cv_data):
        """Save processed CV data to storage"""
        try:
            file_path = os.path.join(self.storage_dir, f"{cv_id}.json")
            with open(file_path, "w") as f:
                json.dump(cv_data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving CV {cv_id}: {str(e)}")
            return False

    def get_cv(self, cv_id):
        """Retrieve a specific CV by ID"""
        try:
            file_path = os.path.join(self.storage_dir, f"{cv_id}.json")
            if not os.path.exists(file_path):
                return None

            with open(file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error retrieving CV {cv_id}: {str(e)}")
            return None

    def get_all_cvs(self):
        """Retrieve all CVs in the database"""
        try:
            cv_files = [f for f in os.listdir(self.storage_dir) if f.endswith(".json")]
            all_cvs = {}

            for file_name in cv_files:
                cv_id = os.path.splitext(file_name)[0]
                cv_data = self.get_cv(cv_id)
                if cv_data:
                    all_cvs[cv_id] = cv_data

            return all_cvs
        except Exception as e:
            logger.error(f"Error retrieving all CVs: {str(e)}")
            return {}

    def delete_cv(self, cv_id):
        """Delete a CV from storage"""
        try:
            file_path = os.path.join(self.storage_dir, f"{cv_id}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting CV {cv_id}: {str(e)}")
            return False
