import tempfile
import os

class TempFileManager:
    def __init__(self, suffix=""):
        self.suffix = suffix
        self.file_name = None

    def __enter__(self):
        self.file_name = tempfile.NamedTemporaryFile(suffix=self.suffix, delete=False).name
        return self.file_name

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file_name and os.path.exists(self.file_name):
            os.remove(self.file_name)
