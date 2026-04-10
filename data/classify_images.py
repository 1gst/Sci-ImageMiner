"""
遍历指定文件夹，找到名为 images 的子文件夹，
解析其中的 ground truth JSON 文件，将每张图片的子图分类信息汇总到一张表中。

用法:
    python classify_images.py --root /path/to/dataset
    python classify_images.py --root /path/to/dataset --output result.csv
"""

import argparse
import glob
import json
import os

import pandas as pd


def find_images_dirs(root_dir: str) -> list:
    """递归查找所有名为 images 的子文件夹"""
    images_dirs = []
    for dirpath, dirnames, _ in os.walk(root_dir):
        if os.path.basename(dirpath) == "images":
            images_dirs.append(dirpath)
    return images_dirs


def parse_ground_truth(images_dir: str) -> list:
    """解析 images 目录下所有 JSON ground truth 文件，提取子图分类信息

    Args:
        images_dir: images 文件夹路径

    Returns:
        包含 (图片路径, 子图, 子图类型, 子图数量) 的记录列表
    """
    records = []
    json_files = glob.glob(os.path.join(images_dir, "*.json"))

    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"跳过无法解析的文件: {json_path}, 错误: {e}")
            continue

        classification = data.get("classification", {})
        if not classification:
            continue

        # 找到与该 JSON 同名的图片文件
        json_basename = os.path.splitext(os.path.basename(json_path))[0]
        image_path = find_matching_image(images_dir, json_basename)

        subfigure_count = len(classification)

        for subfigure_label, subfigure_type in classification.items():
            records.append({
                "图片路径": image_path or json_basename,
                "子图": subfigure_label,
                "子图类型": subfigure_type,
                "子图数量": subfigure_count,
            })

    return records


def find_matching_image(directory: str, basename: str) -> str:
    """在目录中查找与给定 basename 匹配的图片文件"""
    image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp", ".svg"]
    for ext in image_extensions:
        candidate = os.path.join(directory, basename + ext)
        if os.path.exists(candidate):
            return candidate
    return ""


def main():
    parser = argparse.ArgumentParser(description="解析 images 目录下的 ground truth JSON，汇总子图分类信息")
    parser.add_argument("--root", type=str, default="/Users/lingzhou/Resource/DataResource/ALD-E-ImageMiner-main/icdar2026-competition-data/dev", help="要递归搜索的根目录路径")
    parser.add_argument("--output", type=str, default="image_classification_dev_summary.csv", help="输出 CSV 文件路径")
    args = parser.parse_args()

    root_dir = args.root
    if not os.path.isdir(root_dir):
        print(f"错误: 目录不存在: {root_dir}")
        return

    # 递归查找所有 images 文件夹
    images_dirs = find_images_dirs(root_dir)
    if not images_dirs:
        print(f"在 {root_dir} 下未找到名为 'images' 的子文件夹")
        return

    print(f"找到 {len(images_dirs)} 个 images 目录:")
    for directory in images_dirs:
        print(f"  - {directory}")

    # 解析所有 images 目录
    all_records = []
    for images_dir in images_dirs:
        records = parse_ground_truth(images_dir)
        all_records.extend(records)

    if not all_records:
        print("未找到任何有效的分类数据")
        return

    # 生成汇总表
    dataframe = pd.DataFrame(all_records, columns=["图片路径", "子图", "子图类型", "子图数量"])

    # 输出统计信息
    print(f"\n共解析 {len(dataframe)} 条子图记录")
    print(f"涉及 {dataframe['图片路径'].nunique()} 张图片")
    print(f"\n子图类型分布:")
    type_counts = dataframe["子图类型"].value_counts()
    for figure_type, count in type_counts.items():
        percentage = count / len(dataframe) * 100
        print(f"  {figure_type}: {count} ({percentage:.1f}%)")

    # 保存结果
    dataframe.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
