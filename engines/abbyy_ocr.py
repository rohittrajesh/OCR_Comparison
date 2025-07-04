# engines/abbyy_ocr.py

import requests

class Engine:
    def __init__(self, endpoint, application_id, password, profile=None):
        self.endpoint = endpoint.rstrip("/") + "/v2/recognize"
        self.auth = (application_id, password)
        self.profile = profile or "documentArchiving"

    def run(self, path, kind=None):
        """
        Send file to ABBYY Cloud OCR SDK and return a dict with key "text".
        """
        with open(path, "rb") as f:
            files = {"file": f}
            data = {"profile": self.profile}
            resp = requests.post(self.endpoint, auth=self.auth, files=files, data=data)
            resp.raise_for_status()
            js = resp.json()

        # parse ABBYY structure
        lines = []
        for page in js.get("recognitionResults", []):
            for line in page.get("lines", []):
                lines.append(line.get("text", ""))
        return {"text": "\n".join(lines)}
