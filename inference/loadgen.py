"""Load generator for an OpenAI-compatible completions endpoint. Standard library only.

  CONCURRENCY   parallel clients                  (default 8)
  DURATION      seconds to run                    (default 120)
  PROMPT_WORDS  words per prompt                  (default 400)
  MAX_TOKENS    tokens generated per request      (default 256)
  URL           endpoint (default http://vllm:8000/v1/completions)
"""
import json
import os
import random
import statistics
import threading
import time
import urllib.request

URL = os.environ.get("URL", "http://vllm:8000/v1/completions")
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
CONCURRENCY = int(os.environ.get("CONCURRENCY", 8))
DURATION = int(os.environ.get("DURATION", 120))
PROMPT_WORDS = int(os.environ.get("PROMPT_WORDS", 400))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 256))

WORDS = ("the gpu fleet runs training and inference jobs while engineers watch power "
         "temperature clocks memory and network health across every rack").split()

latencies, errors, tokens = [], [], [0]
lock = threading.Lock()
stop_at = time.time() + DURATION


def client():
    while time.time() < stop_at:
        prompt = " ".join(random.choices(WORDS, k=PROMPT_WORDS))
        body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": MAX_TOKENS,
                           "temperature": 0.8, "ignore_eos": True}).encode()
        req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
        t = time.time()
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                out = json.load(r)
            with lock:
                latencies.append(time.time() - t)
                tokens[0] += out["usage"]["completion_tokens"]
        except Exception as e:  # count it and keep going, like a real client with retries
            with lock:
                errors.append(type(e).__name__)
            time.sleep(1)


start = time.time()
threads = [threading.Thread(target=client) for _ in range(CONCURRENCY)]
for t in threads:
    t.start()
for t in threads:
    t.join()
elapsed = time.time() - start

print(f"concurrency={CONCURRENCY} duration={elapsed:.0f}s requests={len(latencies)} errors={len(errors)}")
if errors:
    print("error types:", {e: errors.count(e) for e in set(errors)})
if len(latencies) >= 2:
    q = statistics.quantiles(latencies, n=20)
    print(f"output tokens/s={tokens[0]/elapsed:.0f}  "
          f"latency p50={statistics.median(latencies):.1f}s  p95={q[18]:.1f}s  max={max(latencies):.1f}s")
