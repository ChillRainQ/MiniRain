# collect_zh.py
"""
wukong 中文图文对收集脚本（纯 Python，不走 streaming，避开 Windows 崩溃）
用法：
    python collect_zh.py --num_samples 2000        # 先验证
    python collect_zh.py --num_samples 500000      # 放量
"""
import argparse
import io
import json
import tarfile
from pathlib import Path

from modelscope import dataset_snapshot_download
import pyarrow.parquet as pq
from PIL import Image


def keep(caption: str) -> bool:
    if not (6 <= len(caption) <= 80):
        return False
    bad = ["点击", "购买", "http", "www.", "IMG_", ".jpg", ".png",
           "价格", "淘宝", "链接", "关注", "扫码", "广告"]
    if any(b in caption for b in bad):
        return False
    zh = sum('\u4e00' <= c <= '\u9fff' for c in caption)
    return zh / len(caption) > 0.5


def iter_parquet(pq_file: Path):
    """逐条产出 (caption, image_bytes)。"""
    table = pq.read_table(pq_file)
    cols = table.column_names
    # 字段名兼容：caption/text，image/img
    cap_col = next((c for c in cols if c in ("caption", "text", "txt")), None)
    img_col = next((c for c in cols if c in ("image", "img", "img_bytes")), None)
    if cap_col is None or img_col is None:
        print(f"[跳过] {pq_file.name} 字段不符: {cols}")
        return
    for caption, image in zip(table[cap_col].to_pylist(), table[img_col].to_pylist()):
        raw = image["bytes"] if isinstance(image, dict) else image
        yield caption, raw


def iter_tar(tar_file: Path):
    """wukong 若是 tar 分片（json+jpg 成对），逐条产出 (caption, image_bytes)。"""
    with tarfile.open(tar_file) as tf:
        members = {}
        for m in tf:
            if not m.isfile():
                continue
            stem, ext = m.name.rsplit(".", 1) if "." in m.name else (m.name, "")
            members.setdefault(stem, {})[ext.lower()] = m
        for stem, pair in members.items():
            cap_m = next((pair[e] for e in ("json", "txt") if e in pair), None)
            img_m = next((pair[e] for e in ("jpg", "jpeg", "png", "webp") if e in pair), None)
            if cap_m is None or img_m is None:
                continue
            raw_cap = tf.extractfile(cap_m).read()
            if cap_m.name.endswith(".json"):
                caption = json.loads(raw_cap.decode("utf-8")).get("caption", "")
            else:
                caption = raw_cap.decode("utf-8").strip()
            yield caption, tf.extractfile(img_m).read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=2_000,
                        help="目标条数（清洗后的实际留存）")
    parser.add_argument("--raw_dir", type=str, default="wukong_raw")
    parser.add_argument("--output", type=str, default="wukong.jsonl")
    parser.add_argument("--image_dir", type=str, default="wukong_images")
    args = parser.parse_args()

    # ---------- 1. 下载（数据集仓库，必须 dataset_snapshot_download） ----------
    print("下载 wukong 原始数据...")
    raw_dir = Path(dataset_snapshot_download(
        "wufaxianshi/wukong",  # ← ModelScope 上的正确 id
        cache_dir=args.raw_dir,
    ))
    files = sorted(p for p in raw_dir.rglob("*")
                   if p.suffix.lower() in (".parquet", ".tar"))
    print(f"原始数据目录: {raw_dir}")
    print(f"发现 {len(files)} 个数据文件: "
          f"{[f.name for f in files[:5]]}{' ...' if len(files) > 5 else ''}")
    if not files:
        all_files = [str(p.relative_to(raw_dir)) for p in raw_dir.rglob("*") if p.is_file()]
        raise RuntimeError(f"没有找到 parquet/tar 数据文件，仓库内容：{all_files[:30]}")

    # ---------- 2. 清洗 + 落盘 ----------
    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    kept, scanned = 0, 0
    done = False
    with open(args.output, 'w', encoding='utf-8') as out:
        for f in files:
            it = iter_parquet(f) if f.suffix == ".parquet" else iter_tar(f)
            for caption, raw in it:
                scanned += 1
                caption = str(caption or "").strip()
                if not keep(caption) or not raw:
                    continue
                try:
                    img = Image.open(io.BytesIO(raw)).convert("RGB")
                except Exception:
                    continue  # 坏图直接丢
                img_path = image_dir / f"{kept:07d}.jpg"
                img.save(img_path, "JPEG", quality=95)
                out.write(json.dumps(
                    {"image": str(img_path), "text": caption},
                    ensure_ascii=False) + '\n')
                kept += 1
                if kept % 1000 == 0:
                    print(f"已存 {kept} 条（扫描 {scanned}，留存率 {kept/scanned:.1%}）")
                if kept >= args.num_samples:
                    done = True
                    break
            if done:
                break

    rate = f"{kept/scanned:.1%}" if scanned else "N/A"
    print(f"完成：{kept} 条 → {args.output}，总扫描 {scanned}，留存率 {rate}")


if __name__ == "__main__":
    main()