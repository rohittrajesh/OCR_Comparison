# engines/textract_ocr.py

import boto3
from pdf2image import convert_from_path
import io

class Engine:
    def __init__(self, region, access_key, secret_key):
        self.client = boto3.client(
            "textract",
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )

    def run(self, filepath, kind=None):
        """
        - If filepath ends in .pdf: rasterize each page at 300 DPI.
        - For each page-image, call analyze_document.
        - Collect all the LINE blocks into one big text blob.
        """
        # 1) Get list of PIL images
        if filepath.lower().endswith(".pdf"):
            pages = convert_from_path(filepath, dpi=300)
        else:
            # wrap a single image file path into a list
            from PIL import Image
            pages = [Image.open(filepath)]

        all_lines = []
        for img in pages:
            # 2) Convert PIL image to PNG bytes
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            image_bytes = buf.getvalue()

            # 3) Call synchronous analyze_document
            resp = self.client.analyze_document(
                Document={"Bytes": image_bytes},
                FeatureTypes=["TABLES", "FORMS"]
            )

            # 4) Extract all LINE text
            for blk in resp["Blocks"]:
                if blk["BlockType"] == "LINE":
                    all_lines.append(blk["Text"])

        # 5) Return unified text
        return {"output": "\n".join(all_lines)}
