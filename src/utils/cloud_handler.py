import os
import shutil
import json
import time
from pathlib import Path
from PIL import Image

try:
    import boto3
    from botocore.exceptions import NoCredentialsError
except ImportError:
    boto3 = None

class CloudManager:
    def __init__(self):
        self.bucket_name = os.environ.get("AWS_BUCKET_NAME", "omni-scribe-data")
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY")
        ) if boto3 else None
        
        self.local_cloud_path = Path("data/cloud_storage_simulation")
        self.local_cloud_path.mkdir(parents=True, exist_ok=True)

    def upload_file(self, file_path, object_name=None):
        if object_name is None:
            object_name = os.path.basename(file_path)

        if self.s3_client:
            try:
                print(f"☁️ Attempting S3 Upload: {object_name}")
                self.s3_client.upload_file(file_path, self.bucket_name, object_name)
                return True, f"✅ Uploaded to AWS S3 ({self.bucket_name}/{object_name})"
            except (NoCredentialsError, Exception) as e:
                print(f"⚠️ S3 Upload Failed: {e}")
        
        dest_path = self.local_cloud_path / object_name
        try:
            shutil.copy2(file_path, dest_path)
            return True, f"📂 S3 Unavailable. Saved to Local Cloud: {dest_path}"
        except Exception as e:
            return False, f"❌ Cloud Error: {str(e)}"

    def upload_json_data(self, data_dict, filename):
        """
        Cleanly saves dictionary to JSON by removing non-serializable objects (Images).
        """
        # --- FIX: Create a clean copy without images ---
        clean_data = {}
        for key, value in data_dict.items():
            # Skip keys that hold Image objects
            if isinstance(value, (Image.Image, list)): 
                # Note: 'pages' is a list of images, so we skip lists containing images too
                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], Image.Image):
                    continue
                if isinstance(value, Image.Image):
                    continue
            
            clean_data[key] = value
        # -----------------------------------------------

        temp_path = f"data/temp_{filename}"
        try:
            with open(temp_path, "w") as f:
                json.dump(clean_data, f, indent=4, default=str) # default=str handles dates/other weird types
            
            success, msg = self.upload_file(temp_path, filename)
        except Exception as e:
            return False, f"❌ JSON Save Error: {str(e)}"
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
        return success, msg