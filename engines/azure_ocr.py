import requests, time

class Engine:
    def __init__(self, endpoint, key):
        # Make sure endpoint ends without a slash
        self.url = endpoint.rstrip("/") + "/vision/v3.2/read/analyze"
        self.headers = {
            "Ocp-Apim-Subscription-Key": key,
            "Content-Type": "application/octet-stream"
        }

    def run(self, filepath, kind=None):
        # 1) send the file
        with open(filepath, "rb") as f:
            data = f.read()
        resp = requests.post(self.url, headers=self.headers, data=data)

        # 2) check for errors
        if resp.status_code != 202:
            raise RuntimeError(
                f"Azure OCR POST failed: {resp.status_code} {resp.text}"
            )

        # 3) get the Operation-Location (case-insensitive)
        op_loc = resp.headers.get("Operation-Location") or resp.headers.get("operation-location")
        if not op_loc:
            # dump headers for debugging
            raise RuntimeError(
                f"Azure OCR missing Operation-Location header:\n{resp.headers}"
            )

        # 4) poll until done
        while True:
            poll = requests.get(op_loc, headers={"Ocp-Apim-Subscription-Key": self.headers["Ocp-Apim-Subscription-Key"]})
            j = poll.json()
            if j.get("status") == "succeeded":
                lines = [
                    ln["text"]
                    for pg in j["analyzeResult"]["readResults"]
                    for ln in pg["lines"]
                ]
                return {"output": "\n".join(lines)}
            if j.get("status") == "failed":
                raise RuntimeError(f"Azure OCR failed: {j}")
            time.sleep(0.5)
