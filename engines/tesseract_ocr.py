# engines/tesseract_ocr.py

import subprocess, tempfile, os
# subprocess -- to invoke the Tesseract command-line tool.
# tempfile -- to create temporary files for each page image.
# os -- to delete those temporary files afterward.
from pdf2image import convert_from_path
from PIL import Image

class Engine:
    def __init__(self, langs="eng", tesseract_config="11"):
    # The Tesseract language code(s) (e.g. "eng" for English, "eng+fra" for English + French).
    # The Page Segmentation Mode (PSM) flag (an integer string) Tesseract uses to decide how to split the page into text regions.
    # These get stored on the instance for use during each OCR run. 
        # The integer 3 stands for "Mode 3" -- Fully automatic page segmentation, but no OSD (Orientation and Script Detection
        self.langs = langs
        self.psm   = tesseract_config

    def run(self, filepath, kind=None):
        # 1) Load pages: convert PDF → PIL images, or open single-image files
        if filepath.lower().endswith(".pdf"):
            pages = convert_from_path(filepath, dpi=300)
        else:
            pages = [Image.open(filepath)]

        all_text = []
        # 2) Loop over each PIL image in `pages`
        for img in pages:
            # write image to a temp PNG
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                img.save(tf.name, format="PNG")
                tmp_path = tf.name

            txt_path = tmp_path + ".txt"
            # call tesseract CLI correctly
            cmd = [
                "tesseract",
                tmp_path,
                tmp_path,        # output base name; .txt is appended
                "-l", self.langs,
                "--psm", self.psm
            ]
            # The above is a command array 
            subprocess.run(cmd, check=True)

            # read the OCR result
            with open(txt_path, "r", encoding="utf-8") as f:
                all_text.append(f.read())

            # clean up temp files
            os.remove(tmp_path)
            os.remove(txt_path)

        return {"output": "\n".join(all_text)}
