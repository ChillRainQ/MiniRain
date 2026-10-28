# img_data_downloader.py
"""
图文对收集：MUGE + Flickr30k-CN + COCO-CN + ALLaVA-Caption-LAION（可选中文版）+ ALLaVA-Pretrain-Parquet + LSVT(OCR)

产物（OUT_DIR 下）：
  images/{source}/{image_id}.jpg
  pairs_{source}.jsonl     {"image": 相对路径, "caption": ..., "source": ..., "split": ...}
  stats.json

用法：
  python img_data_downloader.py --only flickr
  python img_data_downloader.py --only muge
  python img_data_downloader.py --only cococn
  python img_data_downloader.py --only allava
  python img_data_downloader.py --only allava --allava_zh_json /path/中文版caption.json
  python img_data_downloader.py --only allava_parquet          # 推荐：预训练用，图已 resize 到 256x256
  python img_data_downloader.py --only lsvt --lsvt_dir /path/LSVT
  全部断点续跑（.done 标记），--keep_files 保留中间文件

注意：
  allava 与 allava_parquet 图像来源高度重叠（同为 LLaVA/LAION 图），二者应二选一。
  默认 --only "" 全跑时不会触发 allava_parquet，需显式指定。
"""
import argparse
import base64
import hashlib
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

# 中文 caption 三源（Chinese-CLIP 预处理包，HF 镜像直链）
OSS_SOURCES = {
    "muge": {
        "urls": [
            "https://hf-mirror.com/datasets/OFA-Sys/chinese-clip-eval/resolve/main/MUGE.zip",
            "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/datasets/MUGE.zip",
        ],
        "splits": ["train", "valid"],
        "min_caption_len": 4, "max_caption_len": 100,
    },
    "flickr": {
        "urls": [
            "https://hf-mirror.com/datasets/OFA-Sys/chinese-clip-eval/resolve/main/Flickr30k-CN.zip",
            "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/datasets/Flickr30k-CN.zip",
        ],
        "splits": ["train", "valid"],
        "min_caption_len": 8, "max_caption_len": 120,
    },
    "cococn": {
        "urls": [
            "https://hf-mirror.com/datasets/OFA-Sys/chinese-clip-eval/resolve/main/COCO-CN.zip",
        ],
        "splits": ["train", "valid"],
        "min_caption_len": 8, "max_caption_len": 120,
    },
}

# ALLaVA：caption json 在 ALLaVA-4V 仓库；图片在 LLaVA 的 LAION-CC-SBU-558K 图包
ALLAVA_CAPTION_URL = (f"{MIRROR}/datasets/FreedomIntelligence/ALLaVA-4V/resolve/main/"
                      f"allava_laion/ALLaVA-Caption-LAION-4V.json")
ALLAVA_IMAGE_URLS = [
    f"{MIRROR}/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip",
    f"{MIRROR}/datasets/AI-ModelScope/LLaVA-Pretrain/resolve/main/images.zip",
]

# ALLaVA-4V 预训练 Parquet（minimind 整理版）：conversations + image_bytes，图像已 resize 到 256x256
ALLAVA_PARQUET_URLS = [
    f"{MIRROR}/datasets/jingyaogong/minimind-v_dataset/resolve/main/pretrain_i2t.parquet",
    "https://huggingface.co/datasets/jingyaogong/minimind-v_dataset/resolve/main/pretrain_i2t.parquet",
]


def download(urls, dst: Path):
    """多候选 URL + Range 断点续传。"""
    if isinstance(urls, str):
        urls = [urls]
    dst.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for url in urls:
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
                                      f"({got/max(time.time()-t0,1e-9)/1e6:.1f} MB/s)",
                                      end="", flush=True)
                    print()
                if total and dst.stat().st_size < total:
                    raise RuntimeError("大小不符，续传重试")
                return
            except Exception as e:
                last_err = e
                print(f"\n  [{url}] 中断: {e}，{2**attempt}s 后重试 ({attempt+1}/5)")
                time.sleep(2 ** attempt)
                mode = "ab"
        dst.unlink(missing_ok=True)   # 这个 URL 彻底失败，换下一个候选源
        mode = "wb"
    raise RuntimeError(f"下载失败: {urls[-1]} (最后错误: {last_err})")


def write_pair(out_f, seen, img_rel, caption, source, split):
    key = (str(img_rel), caption)
    if key in seen:
        return 0
    seen.add(key)
    out_f.write(json.dumps({"image": str(img_rel), "caption": caption,
                            "source": source, "split": split}, ensure_ascii=False) + "\n")
    return 1


# ================= MUGE / Flickr / COCO-CN =================
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
        download(cfg["urls"], zip_path)
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
            print(f"[{name}:{split}] 图片 {n_img} 张，坏图累计 {stats['bad_img']}")
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
        download(ALLAVA_CAPTION_URL, cap_path)
    if not img_zip.exists() and not img_dir.exists():
        download(ALLAVA_IMAGE_URLS, img_zip)
    if not img_dir.exists():
        print("[allava] 解压 images.zip ...")
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

    img_root = None
    for cand in [img_dir / "allava_laion" / "images", img_dir / "images", img_dir]:
        if cand.exists() and any(cand.iterdir()):
            img_root = cand
            break
    assert img_root is not None, "找不到解压后的图片目录"

    # 建一次文件索引 basename -> 路径（含子目录一层兜底）
    idx = {p.name: p for p in img_root.rglob("*") if p.is_file()}
    print(f"[allava] 图片索引 {len(idx)} 个文件")

    img_out = OUT_DIR / "images" / name
    img_out.mkdir(parents=True, exist_ok=True)
    stats = {"images": 0, "pairs": 0, "missing_img": 0, "zh_used": 0}
    seen = set()
    with open(OUT_DIR / f"pairs_{name}.jsonl", "w", encoding="utf-8") as out_f:
        for i, d in enumerate(records):
            fname = Path(d["image"]).name
            stem = Path(fname).stem
            # LLaVA 图包命名：LAION id 补零到 9 位，按前 5 位分目录
            z = stem.zfill(9)
            src = (idx.get(fname) or idx.get(stem + ".jpeg")
                   or idx.get(stem + ".jpg") or idx.get(stem)
                   or idx.get(z + ".jpg") or idx.get(z))
            if src is None:
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
            dst = img_out / (stem + ".jpg")
            if not dst.exists():
                shutil.copyfile(src, dst)
            stats["images"] += 1
            stats["pairs"] += write_pair(out_f, seen, dst.relative_to(OUT_DIR),
                                         caption, name, "train")
            if (i + 1) % 2000 == 0:
                print(f"  [allava] 已处理 {i+1}/{len(records)}, "
                      f"缺失 {stats['missing_img']}", flush=True)

    done.touch()
    if not keep_files:
        shutil.rmtree(img_dir, ignore_errors=True)
        img_zip.unlink(missing_ok=True)
    save_stats(name, stats)


# ================= ALLaVA Pretrain Parquet（minimind 版） =================
def _extract_caption(conv) -> str:
    """兼容 from/value 与 role/content 两种对话格式，取最后一条 assistant 回答。"""
    try:
        msgs = json.loads(conv) if isinstance(conv, str) else conv
    except (json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(msgs, list):
        return ""
    cap = ""
    for m in msgs:
        if not isinstance(m, dict):
            continue
        if m.get("from") == "gpt" or m.get("role") == "assistant":
            cap = (m.get("value") or m.get("content") or "").strip()
    return cap


def process_allava_parquet(keep_files: bool, min_len: int = 8, max_len: int = 2000):
    """从 minimind 整理的 pretrain_i2t.parquet 抽取图像与描述。

    - 图像为 256x256 jpg（已 resize），与 patch_size=16 对齐
    - 以 image_bytes 的 sha1 前 20 位命名，天然去重；不同 caption 保留为多行 pair
    - parquet 原文件不删除（是唯一源头，且下载体积大），keep_files 在此分支无操作
    """
    name = "allava_parquet"
    done = OUT_DIR / f".{name}.done"
    if done.exists():
        print(f"[{name}] 已完成，跳过")
        return

    raw = DL_DIR / name
    pq_path = raw / "pretrain_i2t.parquet"
    if not pq_path.exists():
        download(ALLAVA_PARQUET_URLS, pq_path)

    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise SystemExit(f"[{name}] 需要 pyarrow: pip install pyarrow")

    img_out = OUT_DIR / "images" / name
    img_out.mkdir(parents=True, exist_ok=True)
    pairs_path = OUT_DIR / f"pairs_{name}.jsonl"

    pf = pq.ParquetFile(pq_path)
    cols = pf.schema_arrow.names
    img_col = "image_bytes" if "image_bytes" in cols else ("image" if "image" in cols else None)
    conv_col = "conversations" if "conversations" in cols else ("conversation" if "conversation" in cols else None)
    assert img_col and conv_col, f"[{name}] 列不匹配，可用列: {cols}"
    total_rows = pf.metadata.num_rows
    print(f"[{name}] rows={total_rows}, image_col={img_col}, conv_col={conv_col}")

    stats = {"images": 0, "pairs": 0, "filtered": 0, "bad_img": 0}
    seen = set()
    i = 0
    with open(pairs_path, "w", encoding="utf-8") as out_f:
        for batch in pf.iter_batches(batch_size=1024, columns=[conv_col, img_col]):
            names = batch.schema.names
            convs = batch.column(names.index(conv_col)).to_pylist()
            imgs = batch.column(names.index(img_col)).to_pylist()
            for conv, b in zip(convs, imgs):
                i += 1
                if not b or len(b) < 500:
                    stats["bad_img"] += 1
                    continue
                caption = _extract_caption(conv)
                if not (min_len <= len(caption) <= max_len):
                    stats["filtered"] += 1
                    continue
                h = hashlib.sha1(b).hexdigest()[:20]
                dst = img_out / f"{h}.jpg"
                if not dst.exists():
                    dst.write_bytes(b)
                    stats["images"] += 1
                stats["pairs"] += write_pair(out_f, seen, dst.relative_to(OUT_DIR),
                                             caption, name, "train")
                if i % 20000 == 0:
                    print(f"  [{name}] {i}/{total_rows} "
                          f"images={stats['images']} pairs={stats['pairs']}", flush=True)

    done.touch()
    save_stats(name, stats)


# ================= LSVT（OCR，弱标注全文转写） =================
def process_lsvt(lsvt_dir: str):
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

    img_dir = root / "images_full"
    if not img_dir.exists():
        img_dir.mkdir(parents=True, exist_ok=True)
        for tar in sorted(root.glob("train_full_images_*.tar.gz")):
            print(f"[lsvt] 解包 {tar.name} ...")
            with tarfile.open(tar) as t:
                t.extractall(img_dir)

    print("[lsvt] 解析标注 json ...")
    data = json.load(open(labels, encoding="utf-8"))
    templates = [
        "图中写了什么文字？图中的文字是：{t}",
        "请读出图片中的文字。{t}",
        "这张图片上的文字内容是什么？答案是：{t}",
        "识别图中的文字：{t}",
    ]
    stats = {"images": 0, "pairs": 0, "skipped": 0}
    seen = set()
    with open(OUT_DIR / f"pairs_{name}.jsonl", "w", encoding="utf-8") as out_f:
        for i, (img_name, annos) in enumerate(data.items()):
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
    ap.add_argument("--only", default="",
                    choices=["", "muge", "flickr", "cococn", "allava",
                             "allava_parquet", "lsvt"])
    ap.add_argument("--keep_files", action="store_true")
    ap.add_argument("--allava_zh_json", default="",
                    help="ALLaVA 中文翻译版 caption json（ModelScope 下载）")
    ap.add_argument("--lsvt_dir", default="", help="LSVT 数据目录")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DL_DIR.mkdir(parents=True, exist_ok=True)

    if args.only in ("", "muge", "flickr", "cococn"):
        for name, cfg in OSS_SOURCES.items():
            if args.only and name != args.only:
                continue
            process_oss(name, cfg, args.keep_files)
    if args.only in ("", "allava"):
        process_allava(args.keep_files, args.allava_zh_json)
    if args.only == "allava_parquet":
        # 显式触发：默认全跑不执行，避免与上面的 allava（JSON+images.zip）重复
        process_allava_parquet(args.keep_files)
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