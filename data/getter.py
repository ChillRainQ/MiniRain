# download_data.py
from modelscope.msdatasets import MsDataset
import argparse
import json

def keep(caption):
    if not (6 <= len(caption) <= 80):
        return False
    bad = ["点击", "购买", "http", "www.", "IMG_", ".jpg", ".png",
           "价格", "淘宝", "链接", "关注", "扫码", "广告"]
    if any(b in caption for b in bad):
        return False
    zh = sum('\u4e00' <= c <= '\u9fff' for c in caption)
    return zh / len(caption) > 0.5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=2_000, help="目标条数（清洗后的实际留存）")
    parser.add_argument("--output", type=str, default="wukong_2m.jsonl")
    parser.add_argument("--dataset", type=str, default="AI-ModelScope/wukong")
    args = parser.parse_args()

    ds = MsDataset.load(args.dataset, split='train', streaming=True)

    kept, scanned = 0, 0
    with open(args.output, 'w', encoding='utf-8') as out:
        for sample in ds:
            scanned += 1
            caption = str(sample.get('caption', '')).strip()
            if keep(caption):
                out.write(json.dumps({
                    "image": sample['image'],
                    "text": caption
                }, ensure_ascii=False) + '\n')
                kept += 1
                if kept % 1000 == 0:
                    print(f"已存 {kept} 条（扫描 {scanned}，留存率 {kept/scanned:.1%}）")
                if kept >= args.num_samples:
                    break
    print(f"完成：{kept} 条 → {args.output}，总扫描 {scanned}，留存率 {kept/scanned:.1%}")