# collect_data.py
"""
图文对收集：MUGE + Flickr30k-CN + ALLaVA-Caption-LAION（可选中文翻译版）+ LSVT(OCR)
训练配比（在训练采样控制，不在此脚本）：MUGE/Flickr ×2 epoch，ALLaVA ×1，OCR 若干

产物（OUT_DIR 下）：
  images/{source}/{image_id}.jpg
  pairs_{source}.jsonl     {"image": 相对路径, "caption": ..., "source": ..., "split": ...}
  stats.json

用法：
  python collect_data.py --only flickr          # 先小跑验证
  python collect_data.py --only muge
  python collect_data.py --only allava          # 英文版 caption
  python collect_data.py --only allava --allava_zh_json /path/中文版.json   # 用中文翻译
  python collect_data.py --only lsvt --lsvt_dir /path/LSVT                  # 需先备好 LSVT 文件
  全部断点续跑，.done 标记控制；--keep_files 保留中间文件
"""
import argparse
import base64
import json
import shutil
import time
import zipfile
from pathlib import Path

import requests

OUT_DIR = Path("/data/xu/xin/MiniRain/data/image_text")
DL_DIR = OUT_DIR / "_downloads"
MIRROR = "https://hf-mirror.com"
CHUNK = 1 << 20

OSS_SOURCES = {
    "muge": {
        "url": "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/datasets/MUGE.zip",
        "splits": ["train", "valid"],
        "min_caption_len": 4, "max_caption_len": 100,
    },
    "flickr": {
        "url": "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/datasets/Flickr30k-CN.zip",
        "splits": ["train", "valid"],
        "min_caption_len": 8, "max_caption_len": 120,
    },
}

# ALLaVA LAION：caption json + 官方 images.zip（HF 镜像直链）
ALLAVA_REPO = "FreedomIntelligence/ALLaVA-4V"
ALLAVA_FILES = {
    "caption": "allava_laion/ALLaVA-Caption-LAION-4V.json",
    "images": "allava_laion/images.zip",
}


def download(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    headers, mode = {}, "wb"
    if dst.exists():
        headers["Range"] = f"bytes={dst.stat().st_size}-"
        mode = "ab"
    for attempt in range(5):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status_code}")
                cl = r.headers.get("Content-Length")
                base = dst.stat().st_size if mode == "ab" else 0
                total = base + int(cl) if cl else 0
                got, t0 = base, time.time()
                with open(dst, mode) as f:
                    for chunk in r.iter_content(CHUNK):
                        f.write(chunk)
                        got += len(chunk)
                        if total:
                            print(f"\r  {dst.name}: {got/1e6:.0f}/{total/1e6:.0f} MB "
                                  f"({got/max(time.time()-t0,1e-9)/1e6:.1f} MB/s)", end="", flush=True)
                print()
            if total and dst.stat().st_size < total:
                raise RuntimeError("大小不符，续传重试")
            return
        except Exception as e:
            print(f"\n  下载中断: {e}，{2**attempt}s 后重试 ({attempt+1}/5)")
            time.sleep(2 ** attempt)
            mode = "ab"
    raise RuntimeError(f"下载失败: {url}")


def hf_download(repo_file: str, dst: Path):
    download(f"{MIRROR}/datasets/{ALLAVA_REPO}/resolve/main/{repo_file}", dst)


def write_pair(out_f, seen, img_rel, caption, source, split):
    key = (str(img_rel), caption)
    if key in seen:
        return 0
    seen.add(key)
    out_f.write(json.dumps({"image": str(img_rel), "caption": caption,
                            "source": source, "split": split}, ensure_ascii=False) + "\n")
    return 1


# ================= MUGE / Flickr（Chinese-CLIP 预处理包） =================
def find_root(extract_dir: Path, split: str) -> Path:
    for p in extract_dir.rglob(f"{split}_imgs.tsv"):
        return p.parent
    raise FileNotFoundError(f"{split}_imgs.tsv")


def process_oss(name: str, cfg: dict, keep_files: bool):
    zip_path = DL_DIR / f"{name}.zip"
    extract_dir = DL_DIR / f"{name}_raw"
    img_out = OUT_DIR / "images" / name
    pairs_path = OUT_DIR / f"pairs_{name}.jsonl"
    done = OUT_DIR / f".{name}.done"
    if done.exists():
        print(f"[{name}] 已完成，跳过")
        return

    if not zip_path.exists() and not extract_dir.exists():
        download(cfg["url"], zip_path)
    if not extract_dir.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)

    img_out.mkdir(parents=True, exist_ok=True)
    stats = {"images": 0, "pairs": 0, "bad_img": 0, "filtered": 0}
    seen = set()
    with open(pairs_path, "w", encoding="utf-8") as out_f:
        for split in cfg["splits"]:
            mark = OUT_DIR / f".{name}.{split}.done"
            if mark.exists():
                continue
            try:
                root = find_root(extract_dir, split)
            except FileNotFoundError:
                mark.touch()
                continue
            n_img = 0
            with open(root / f"{split}_imgs.tsv", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f):
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) != 2:
                        continue
                    img_id, b64 = parts
                    jpg = img_out / f"{img_id}.jpg"
                    if not jpg.exists():
                        try:
                            data = base64.b64decode(b64)
                            if len(data) < 500:
                                stats["bad_img"] += 1
                                continue
                            jpg.write_bytes(data)
                        except Exception:
                            stats["bad_img"] += 1
                            continue
                    n_img += 1
                    if (i + 1) % 5000 == 0:
                        print(f"  [{name}:{split}] 已解码 {i+1} 行")
            print(f"[{name}:{split}] 图片 {n_img} 张")
            stats["images"] += n_img

            n_pair = 0
            with open(root / f"{split}_texts.jsonl", encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        caption = d["text"].strip()
                    except (json.JSONDecodeError, KeyError):
                        continue
                    if not (cfg["min_caption_len"] <= len(caption) <= cfg["max_caption_len"]):
                        stats["filtered"] += 1
                        continue
                    for img_id in d.get("image_ids", []):
                        jpg = img_out / f"{img_id}.jpg"
                        if jpg.exists():
                            n_pair += write_pair(out_f, seen, jpg.relative_to(OUT_DIR),
                                                 caption, name, split)
            print(f"[{name}:{split}] 图文对 {n_pair} 条")
            stats["pairs"] += n_pair
            mark.touch()

    done.touch()
    if not keep_files:
        shutil.rmtree(extract_dir, ignore_errors=True)
        zip_path.unlink(missing_ok=True)
    save_stats(name, stats)


# ================= ALLaVA-Caption-LAION =================
def process_allava(keep_files: bool, zh_json: str = ""):
    name = "allava"
    done = OUT_DIR / f".{name}.done"
    if done.exists():
        print(f"[{name}] 已完成，跳过")
        return
    raw = DL_DIR / "allava"
    cap_path = raw / "ALLaVA-Caption-LAION-4V.json"
    img_zip = raw / "images.zip"
    img_dir = raw / "images_raw"

    if not cap_path.exists():
        hf_download(ALLAVA_FILES["caption"], cap_path)
    if not img_zip.exists() and not img_dir.exists():
        hf_download(ALLAVA_FILES["images"], img_zip)
    if not img_dir.exists():
        print("[allava] 解压 images.zip（约几十万张，需一段时间）...")
        img_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(img_zip) as z:
            z.extractall(img_dir)

    # 中文翻译版：按 id 建映射，替换 gpt 回答
    zh_map = {}
    if zh_json:
        print(f"[allava] 加载中文翻译 {zh_json}")
        with open(zh_json, encoding="utf-8") as f:
            for d in json.load(f):
                for c in d.get("conversations", []):
                    if c.get("from") == "gpt":
                        zh_map[d["id"]] = c["value"]
        print(f"  中文 caption {len(zh_map)} 条")

    records = json.load(open(cap_path, encoding="utf-8"))
    print(f"[allava] caption 记录 {len(records)} 条")

    # 定位图片根目录（images.zip 内可能带 allava_laion/images 前缀）
    img_root = None
    for cand in [img_dir / "allava_laion" / "images", img_dir / "images", img_dir]:
        if cand.exists() and any(cand.iterdir()):
            img_root = cand
            break
    assert img_root is not None, "找不到解压后的图片目录"

    img_out = OUT_DIR / "images" / name
    img_out.mkdir(parents=True, exist_ok=True)
    stats = {"images": 0, "pairs": 0, "missing_img": 0, "zh_used": 0}
    seen = set()
    with open(OUT_DIR / f"pairs_{name}.jsonl", "w", encoding="utf-8") as out_f:
        for i, d in enumerate(records):
            rel = d["image"]                       # allava_laion/images/xxx.jpeg
            fname = Path(rel).name
            src = img_root / fname
            if not src.exists():
                # 兜底：按相对路径找
                alt = list(img_dir.rglob(fname))
                src = alt[0] if alt else None
            if src is None or not src.exists():
                stats["missing_img"] += 1
                continue
            caption = ""
            for c in d.get("conversations", []):
                if c.get("from") == "gpt":
                    caption = c["value"].strip()
            if not caption:
                continue
            if d["id"] in zh_map:
                caption = zh_map[d["id"]].strip()
                stats["zh_used"] += 1
            dst = img_out / (Path(fname).stem + ".jpg")
            if not dst.exists():
                shutil.copyfile(src, dst)
            stats["images"] += 1
            stats["pairs"] += write_pair(out_f, seen, dst.relative_to(OUT_DIR),
                                         caption, name, "train")
            if (i + 1) % 20000 == 0:
                print(f"  [allava] 已处理 {i+1}/{len(records)}")

    done.touch()
    if not keep_files:
        shutil.rmtree(img_dir, ignore_errors=True)
        img_zip.unlink(missing_ok=True)
    save_stats(name, stats)


# ================= LSVT（OCR，弱标注全文转写） =================
def process_lsvt(lsvt_dir: str):
    """
    需先备好 LSVT 文件（hyper.ai 种子或官方）：
      {lsvt_dir}/train_full_images_0.tar.gz, _1.tar.gz（或已解压的图片目录）
      {lsvt_dir}/train_full_labels.json
    弱标注图每张一句全文转写 → OCR caption + 问法模板
    """
    name = "lsvt"
    done = OUT_DIR / f".{name}.done"
    if done.exists():
        print(f"[{name}] 已完成，跳过")
        return
    root = Path(lsvt_dir)
    labels = root / "train_full_labels.json"
    assert labels.exists(), f"未找到 {labels}，请先下载 LSVT"

    import tarfile
    img_out = OUT_DIR / "images" / name
    img_out.mkdir(parents=True, exist_ok=True)

    # 解出全部图片（若已解压则跳过）
    img_dir = root / "images_full"
    if not img_dir.exists():
        img_dir.mkdir(parents=True, exist_ok=True)
        for tar in sorted(root.glob("train_full_images_*.tar.gz")):
            print(f"[lsvt] 解包 {tar.name} ...")
            with tarfile.open(tar) as t:
                t.extractall(img_dir)

    print("[lsvt] 流式解析标注 json ...")
    data = json.load(open(labels, encoding="utf-8"))
    # 标注结构: {img_name: [{"transcription": ..., "points": ...}, ...]}（若结构不同按实际调整）
    templates = [
        "图中写了什么文字？图中的文字是：{t}",
        "请读出图片中的文字。{t}",
        "这张图片上的文字内容是什么？答案是：{t}",
        "识别图中的文字：{t}",
    ]
    stats = {"images": 0, "pairs": 0, "skipped": 0}
    seen = set()
    items = data.items() if isinstance(data, dict) else []
    with open(OUT_DIR / f"pairs_{name}.jsonl", "w", encoding="utf-8") as out_f:
        for i, (img_name, annos) in enumerate(items):
            texts = [a.get("transcription", "").strip() for a in annos
                     if a.get("transcription", "").strip() and a.get("transcription") != "###"]
            if not texts:
                stats["skipped"] += 1
                continue
            full = " ".join(texts)
            if not (2 <= len(full) <= 200):
                stats["skipped"] += 1
                continue
            hits = list(img_dir.rglob(img_name)) or list(img_dir.rglob(img_name + ".jpg"))
            if not hits:
                stats["skipped"] += 1
                continue
            dst = img_out / (Path(img_name).stem + ".jpg")
            if not dst.exists():
                shutil.copyfile(hits[0], dst)
            stats["images"] += 1
            # 一条直写 + 一条模板问答（问法按图轮转）
            stats["pairs"] += write_pair(out_f, seen, dst.relative_to(OUT_DIR),
                                         full, name, "train")
            stats["pairs"] += write_pair(out_f, seen, dst.relative_to(OUT_DIR),
                                         templates[i % len(templates)].format(t=full),
                                         name + "_vqa", "train")
            if (i + 1) % 20000 == 0:
                print(f"  [lsvt] 已处理 {i+1} 条")
    done.touch()
    save_stats(name, stats)


def save_stats(name, stats):
    p = OUT_DIR / "stats.json"
    all_s = json.loads(p.read_text()) if p.exists() else {}
    all_s[name] = stats
    p.write_text(json.dumps(all_s, ensure_ascii=False, indent=2))
    print(f"[{name}] 完成: {stats}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", choices=["", "muge", "flickr", "allava", "lsvt"])
    ap.add_argument("--keep_files", action="store_true")
    ap.add_argument("--allava_zh_json", default="", help="ALLaVA 中文翻译版 json（ModelScope 下载）")
    ap.add_argument("--lsvt_dir", default="", help="LSVT 数据目录")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DL_DIR.mkdir(parents=True, exist_ok=True)

    if args.only in ("", "muge", "flickr"):
        for name, cfg in OSS_SOURCES.items():
            if args.only and name != args.only:
                continue
            process_oss(name, cfg, args.keep_files)
    if args.only in ("", "allava"):
        process_allava(args.keep_files, args.allava_zh_json)
    if args.only in ("", "lsvt"):
        assert args.lsvt_dir, "LSVT 需要 --lsvt_dir"
        process_lsvt(args.lsvt_dir)

    total = 0
    for p in sorted(OUT_DIR.glob("pairs_*.jsonl")):
        n = sum(1 for _ in open(p, encoding="utf-8"))
        print(f"{p.name}: {n} 条")
        total += n
    print(f"\n总计: {total} 条图文对")


if __name__ == "__main__":
    main()