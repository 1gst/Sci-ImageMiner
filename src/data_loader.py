"""
数据加载和处理的工具类
用于加载Sci-ImageMiner竞赛数据集
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import base64


class SciFigureDataLoader:
    """科学图表数据加载器"""

    def __init__(self, data_root: str):
        """
        初始化数据加载器
        
        Args:
            data_root: 数据集根目录，如 "icdar2026-competition-data"
        """
        self.data_root = Path(data_root)

    def load_data_from_split(self, split: str = "dev") -> List[Dict[str, Any]]:
        """
        加载指定分割的数据集
        
        Args:
            split: 数据集划分，如 "dev", "train", "test"
            
        Returns:
            样本列表，每个样本包含图片路径、标注数据等
        """
        split_path = self.data_root / split
        if not split_path.exists():
            raise ValueError(f"Split path not found: {split_path}")

        samples = []

        # 遍历所有的主类别
        for main_path in split_path.iterdir():
            if not main_path.exists() or not main_path.is_dir():
                continue

            # 遍历所有子类别
            for sub_category in os.listdir(main_path):
                sub_path = main_path / sub_category
                if not sub_path.is_dir():
                    continue

                # 遍历所有论文编号
                for paper_id in os.listdir(sub_path):
                    paper_path = sub_path / paper_id
                    if not paper_path.is_dir():
                        continue

                    # 加载该论文下的所有图片和标注
                    paper_samples = self._load_paper_samples(paper_path, main_path, sub_category, paper_id)
                    samples.extend(paper_samples)

        return samples

    def _load_paper_samples(self, paper_path: Path, main_category: Path,
                            sub_category: str, paper_id: str) -> List[Dict[str, Any]]:
        """
        加载单个论文下的所有样本
        
        Args:
            paper_path: 论文目录路径
            main_category: 主类别
            sub_category: 子类别
            paper_id: 论文ID
            
        Returns:
            该论文下的所有样本列表
        """
        samples = []
        # 遍历路径下文件获取论文pdf的字符串路径
        pdf_path = None
        for file in paper_path.iterdir():
            if file.suffix == '.pdf':
                pdf_path = file
                break
        images_path = paper_path / "images"

        if not images_path.exists():
            return samples

        # 遍历所有JSON标注文件
        for json_file in images_path.glob("*.json"):
            try:
                sample = self._load_single_sample(json_file, main_category, sub_category, paper_id)
                if sample:
                    sample['pdf_path'] = str(pdf_path)
                    samples.append(sample)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue

        return samples

    def _load_single_sample(self, json_path: Path, main_category: Path,
                            sub_category: str, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        加载单个样本
        
        Args:
            json_path: JSON标注文件路径
            main_category: 主类别
            sub_category: 子类别
            paper_id: 论文ID
            
        Returns:
            样本数据字典
        """
        # 读取JSON标注
        with open(json_path, 'r', encoding='utf-8') as f:
            annotation = json.load(f)

        # 找到对应的图片文件
        base_name = json_path.stem
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_path = None

        for ext in image_extensions:
            potential_path = json_path.parent / f"{base_name}{ext}"
            if potential_path.exists():
                image_path = potential_path
                break

        if not image_path:
            print(f"Image not found for JSON: {json_path}")
            return None

        # 构建样本字典
        sample = {
            'sample_id': annotation.get('sample_id', ''),
            'image_path': str(image_path),
            'json_path': str(json_path),
            'annotation': annotation,
            'main_category': main_category.name,
            'sub_category': sub_category,
            'paper_id': paper_id
        }

        return sample

    def encode_image_to_base64(self, image_path: str) -> str:
        """
        将图片编码为base64格式
        
        Args:
            image_path: 图片路径
            
        Returns:
            base64编码的字符串
        """
        with open(image_path, 'rb') as f:
            image_data = f.read()
        return base64.b64encode(image_data).decode('utf-8')

    def get_sample_count(self, split: str = "dev") -> int:
        """
        获取指定分割的样本数量
        
        Args:
            split: 数据集划分
            
        Returns:
            样本数量
        """
        samples = self.load_data_from_split(split)
        return len(samples)

    def filter_samples_by_task(self, samples: List[Dict[str, Any]],
                               task: str) -> List[Dict[str, Any]]:
        """
        根据任务筛选样本
        
        注意: 此方法已不再使用。
        当前实现是对所有样本都进行四个任务推理,对于无法推理的任务返回空结果。
        
        Args:
            samples: 样本列表
            task: 任务名称，如 "classification", "data_extraction", "summarization", "vqa"
            
        Returns:
            筛选后的样本列表
        """
        filtered = []
        for sample in samples:
            annotation = sample['annotation']
            if task in annotation and annotation[task]:
                filtered.append(sample)
        return filtered


def load_captions(content_json_path: str) -> Dict[str, str]:
    """
    加载论文的captions信息
    
    Args:
        content_json_path: content.json文件路径
        
    Returns:
        标签字典：{figure_name: caption_text}
    """
    if not os.path.exists(content_json_path):
        return {}

    with open(content_json_path, 'r', encoding='utf-8') as f:
        contents = json.load(f)

    captions = {}

    # 提取所有图片和fig数据的captions
    # 这个函数需要根据实际的content.json结构进行调整
    for content in contents:
        if isinstance(content, dict) and (content["type"] == 'image' or content["type"] == 'table'):
            if 'img_path' in content:
                captions[content.pop('img_path')] = content
    return captions


if __name__ == '__main__':
    # 测试数据加载器
    loader = SciFigureDataLoader("icdar2026-competition-data")

    print("加载dev数据集...")
    samples = loader.load_data_from_split("dev")
    print(f"总样本数: {len(samples)}")

    # 打印前几个样本的信息
    for i, sample in enumerate(samples[:3]):
        print(f"\n样本 {i + 1}:")
        print(f"  Sample ID: {sample['sample_id']}")
        print(f"  Image Path: {sample['image_path']}")
        print(f"  Main Category: {sample['main_category']}")
        print(f"  Sub Category: {sample['sub_category']}")

        # 打印标注信息
        annotation = sample['annotation']
        print(f"  Classification: {annotation.get('classification', {})}")
        print(f"  Summarization: {annotation.get('summarization', {})}")
        print(f"  Data Extraction: {annotation.get('data_extraction', {})}")

        vqa = annotation.get('vqa', {})
        if vqa:
            print(f"  VQA: {len(vqa)} 个子图的问答对")
