import os
import json
from logging import getLogger
from datetime import datetime

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

    def get_all_cvs(self) -> dict:
        """Retrieve all CVs in the database"""
        try:
            cv_files = [f for f in os.listdir(self.storage_dir) if f.endswith(".json")]
            all_cvs = []

            for file_name in cv_files:
                cv_id = os.path.splitext(file_name)[0]
                cv_data = self.get_cv(cv_id)

                if cv_data:
                    # Extract file meta data
                    meta_info = cv_data.get("meta", {})
                    file_info = {
                        "filename": meta_info.get("filename", ""),
                        "file_size": meta_info.get("file_size", 0),
                        "file_type": meta_info.get("file_type", ""),
                        "upload_date": meta_info.get(
                            "upload_date", str(datetime.utcnow())
                        ),
                    }

                    # Prepare the CV data without meta information
                    cv_data_without_meta = {
                        k: v for k, v in cv_data.items() if k != "meta"
                    }

                    # Add file_info to each CV data in the response
                    cv_data_with_info = {
                        "id": cv_id,
                        "file_info": file_info,
                        **cv_data_without_meta,
                    }

                    # Append CV data to the result list
                    all_cvs.append(cv_data_with_info)

            # Return structured response with meta and data
            return {
                "meta": {
                    "total_count": len(all_cvs),
                },
                "data": all_cvs,
            }
        except Exception as e:
            # Handle errors, e.g., file reading, json parsing errors
            return {"error": str(e)}

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
