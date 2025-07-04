import yaml, json, time, os, re
from concurrent.futures import ThreadPoolExecutor
from engines.azure_ocr import Engine as AzureOCR
from engines.textract_ocr import Engine as TextractOCR
from engines.tesseract_ocr import Engine as TesseractOCR
from engines.o4_llm import Engine as O4LLM
from engines.mistral7b_llm import Engine as Mistral7BLLM
from engines.llama2_llm import Engine as Llama2LLM

def load_cfg():
    return yaml.safe_load(open("config.yaml"))

def make_engines(cfg):
    return {
        # OCR engines
        "AzureOCR":  AzureOCR(**cfg["azure_ocr"]),
        "Textract":  TextractOCR(**cfg["textract_ocr"]),
        "Tesseract": TesseractOCR(**cfg["tesseract_ocr"]),
        # LLM engines
        "o4":        O4LLM(**cfg["o4"]),
        "Mistral7b": Mistral7BLLM(**cfg["mistral7b"]),
        "Llama2":    Llama2LLM(**cfg["llama2"]),
    }

# Read lists of items to benchmark
with open("data/ocr_list.txt") as f:
    OCR_ITEMS = [l.strip() for l in f if l.strip()]
with open("data/llm_list.txt") as f:
    LLM_ITEMS = [l.strip() for l in f if l.strip()]

def extract_text(out):
    """Pull the actual OCR string out of whatever dict your engine returned."""
    if isinstance(out, dict):
        for key in ("text", "ocr_text", "result", "raw_text"):
            v = out.get(key)
            if isinstance(v, str) and v.strip():
                return v
        # fallback to any non-empty string value
        for v in out.values():
            if isinstance(v, str) and v.strip():
                return v
        return ""
    return out or ""

def compute_wer(ref, hyp):
    """Compute WER = (S + D + I) / N via dynamic programming."""
    r = ref.split()
    h = hyp.split()
    n, m = len(r), len(h)

    # Build DP table
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        dp[i][0] = i
    for j in range(1, m+1):
        dp[0][j] = j

    for i in range(1, n+1):
        for j in range(1, m+1):
            if r[i-1] == h[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j]   + 1,  # deletion
                    dp[i][j-1]   + 1,  # insertion
                    dp[i-1][j-1] + 1   # substitution
                )

    # Backtrack to count S, D, I
    subs = dels = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and r[i-1] == h[j-1]:
            i, j = i-1, j-1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            ins += 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            dels += 1; i -= 1
        else:
            subs += 1; i -= 1; j -= 1

    wer = (subs + dels + ins) / float(n) if n > 0 else None
    return wer, subs, dels, ins, n

def normalize(s):
    """Lowercase, strip punctuation (keeping Korean chars), collapse whitespace."""
    s = s.lower()
    # keep word chars, whitespace, and Korean syllables
    s = re.sub(r"[^\w\sㄱ-힣]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def bench(engine_name, engine, item, kind):
    start = time.perf_counter()
    out = engine.run(item, kind=kind)
    latency = time.perf_counter() - start

    rec = {
        "engine": engine_name,
        "item":   item,
        "kind":   kind,
        "latency": latency
    }

    if kind == "ocr":
        text = extract_text(out)
        rec.update({
            "throughput_imgs_per_s":  1.0/latency if latency>0 else None,
            "throughput_chars_per_s": len(text)/latency if latency>0 else None,
            "text": text
        })

        # If a ground-truth .txt exists, normalize & compute WER
        gt = os.path.splitext(item)[0] + ".txt"
        if os.path.exists(gt):
            ref = open(gt, encoding="utf-8").read().strip()
            ref_norm = normalize(ref)
            hyp_norm = normalize(text)
            wer, S, D, I, N = compute_wer(ref_norm, hyp_norm)
            rec.update({
                "wer":               wer,
                "WER_substitutions": S,
                "WER_deletions":     D,
                "WER_insertions":    I,
                "WER_ref_words":     N
            })

    else:  # LLM
        tokens_in  = out.get("tokens_in")  if isinstance(out, dict) else None
        tokens_out = out.get("tokens_out") if isinstance(out, dict) else None
        rec.update({
            "tokens_in":             tokens_in,
            "tokens_out":            tokens_out,
            "throughput_toks_per_s": tokens_out/latency if latency>0 and tokens_out else None,
            "response":              out.get("response") if isinstance(out, dict) else None
        })

    return rec

if __name__ == "__main__":
    cfg     = load_cfg()
    engines = make_engines(cfg)
    perf_recs = []
    ocr_texts = []

    for name, eng in engines.items():
        kind  = "ocr" if name in ("AzureOCR","Textract","Tesseract") else "llm"
        items = OCR_ITEMS if kind=="ocr" else LLM_ITEMS
        for item in items:
            r = bench(name, eng, item, kind)
            if kind == "ocr":
                ocr_texts.append({"engine": name, "item": item, "text": r.pop("text")})
            perf_recs.append(r)

    # Write out two files: performance metrics and raw OCR texts
    json.dump(perf_recs, open("benchmark_results.json","w"), indent=2, ensure_ascii=False)
    json.dump(ocr_texts, open("ocr_texts.json",      "w"), indent=2, ensure_ascii=False)
    print(f"Wrote {len(perf_recs)} records to benchmark_results.json")