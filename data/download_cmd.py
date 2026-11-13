import os
os.environ["HF_ENDPOINT"] = "https://huggingface.co"

import json
from pathlib import Path
from datasets import load_dataset

OUTPUT = Path(__file__).resolve().parent / "ultradata"
OUTPUT.mkdir(parents=True, exist_ok=True)

DATA = OUTPUT / "windows_batchfile.jsonl"
STATE = OUTPUT / "windows_batchfile.state.json"

done = 0
if STATE.exists():
    done = json.loads(STATE.read_text())["rows"]

ds = load_dataset(
    "bigcode/the-stack",
    data_dir="data/batchfile",
    split="train",
    streaming=True,
    token=True,
)

if done:
    ds = ds.skip(done)

with DATA.open("a", encoding="utf-8") as f:
    try:
        for row in ds:
            f.write(json.dumps({
                "text": row["content"],
                "language": "batchfile",
                "source": "bigcode/the-stack",
                "metadata": {
                    k: row.get(k)
                    for k in (
                        "repository_name",
                        "licenses",
                        "hexsha",
                        "ext",
                    )
                },
            }, ensure_ascii=False, default=str) + "\n")
            done += 1

            if done % 1000 == 0:
                f.flush()
                STATE.write_text(json.dumps({"rows": done}))
                print(f"[CMD] {done:,} records", flush=True)
    finally:
        f.flush()
        STATE.write_text(json.dumps({"rows": done}))

print(f"[DONE] {done:,} records")