"""
Sci-ImageMiner 主程序
处理dev数据集并生成四个任务的预测结果
"""

import os
import json
import traceback
import asyncio
import zipfile
import shutil

import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from tqdm import tqdm

from data_loader import SciFigureDataLoader
from task_inferencer import SciFigureInference

# 加载配置
with open("../config/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)["inference"]


class SciImageMinerPipeline:
    """Sci-ImageMiner竞赛处理流水线"""

    def __init__(self):
        """初始化流水线"""
        # 读取API密钥
        api_key_template = config["model"]["api_key"]
        if api_key_template.startswith("${") and api_key_template.endswith("}"):
            env_var = api_key_template[2:-1]
            api_key = os.environ.get(env_var, "")
            if not api_key:
                raise ValueError(f"API key not found in environment variable: {env_var}")
        else:
            api_key = api_key_template

        # 初始化数据加载器
        self.data_loader = SciFigureDataLoader(config["data"]["root"])
        # 配置
        self.split = config["data"]["split"]
        self.tasks = config["tasks"]
        self.output_root = config["output"]["root"]
        # 初始化推理器
        self.inferencer = SciFigureInference(
            model_name=config["model"]["name"],
            base_url=config["model"]["base_url"],
            api_key=api_key,
            proxy_url=config["model"].get("proxy_url", None),
            max_workers=config["concurrency"]["max_workers"],
            config_tasks=self.tasks,
            prompt_versions=config.get("prompt_versions", None)
        )
        # 结果文件名版本标签（按任务独立）
        self.task_version_tags = self.inferencer.task_version_tags

        import aiofiles

        # 为每个结果文件创建异步锁，防止并发写入冲突
        self.file_locks = {}
        for task in self.tasks:
            self.file_locks[task] = asyncio.Lock()

        # 初始化时加载所有任务的 sample_id 集合（用于快速判断任务是否已处理）
        self._load_all_sample_ids()

        print(f"Pipeline initialized with model: {config['model']['name']}")

    async def process_data_set(self) -> Dict[str, str]:
        """
        处理数据集 - 使用异步并发处理（带进度条）
        
        Returns:
            输出文件路径字典 {task: file_path}
        """
        # 创建输出目录
        output_dir = self.output_root
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 加载dev数据集
        print(f"Loading {self.split} dataset from {config['data']['root']}/{self.split}...")
        samples = self.data_loader.load_data_from_split(self.split)
        print(f"Total samples: {len(samples)}")

        # 初始化结果存储
        all_results = {task: [] for task in self.tasks}

        # 异步预筛选未处理的样本
        print("Checking processed samples...")
        unprocessed_samples, processed_count = await self._filter_unprocessed_samples_async(
            samples, output_dir, self.tasks
        )

        if processed_count > 0:
            print(f"Found {processed_count} already processed samples, skipping...")

        if not unprocessed_samples:
            print("All samples have been processed. Nothing to do.")
            # 加载所有已存在的结果
            for task in self.tasks:
                result_file = output_path / f"{task}_results_{self.task_version_tags[task]}.json"
                if result_file.exists():
                    with open(result_file, 'r', encoding='utf-8') as f:
                        all_results[task] = json.load(f)
            output_files = {task: str(output_path / f"{task}_results_{self.task_version_tags[task]}.json") for task in self.tasks}
            return output_files

        print(f"Processing {len(unprocessed_samples)} samples...\n")

        # 批量处理样本 - 使用异步并发（带进度条）
        batch_size = config["concurrency"]["max_workers"]
        with tqdm(total=len(unprocessed_samples), desc="Processing samples", unit="sample") as pbar:
            for i in range(0, len(unprocessed_samples), batch_size):
                batch = unprocessed_samples[i:i + batch_size]

                # 异步处理一个批次（每个样本会立即自动保存结果）
                await self._process_batch(batch, output_dir, pbar)

        # 加载最终结果
        output_files = {}
        for task in self.tasks:
            result_file = output_path / f"{task}_results_{self.task_version_tags[task]}.json"
            if result_file.exists():
                with open(result_file, 'r', encoding='utf-8') as f:
                    all_results[task] = json.load(f)
            output_files[task] = str(result_file)

        print(f"\nAll results saved to {output_dir}")
        return output_files

    async def _process_batch(self, batch: List[Dict[str, Any]],
                             output_dir: str, pbar=None) -> None:
        """
        异步处理一个批次的样本（每个样本处理后立即写入文件）

        Args:
            batch: 样本批次
            output_dir: 输出目录
            pbar: tqdm进度条对象（可选）
        """
        # 为每个样本创建异步任务
        tasks = []
        for sample in batch:
            task = asyncio.create_task(
                self._process_single_sample_async(sample, output_dir, pbar)
            )
            tasks.append(task)

        # 并发执行所有任务，异常通过 return_exceptions 捕获后统一打印
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                if pbar:
                    pbar.write(f"  Error: {result}")
                else:
                    print(f"  Error: {result}")

    async def _process_single_sample_async(self, sample: Dict[str, Any],
                                           output_dir: str, pbar=None) -> None:
        """
        异步处理单个样本，只处理还未处理的任务，并立即保存结果到文件

        Args:
            sample: 样本数据
            output_dir: 输出目录
            pbar: tqdm进度条对象（可选）
        """
        sample_id = sample['sample_id']
        image_path = sample['image_path']
        annotation = sample['annotation']
        pdf_path = sample['pdf_path']

        # 获取已处理的任务列表
        output_path = Path(output_dir)
        existing_tasks = self._get_existing_tasks(sample_id, output_dir, self.tasks)

        # 如果所有任务都已处理，直接跳过
        if len(existing_tasks) == len(self.tasks):
            if pbar:
                pbar.update(1)
            return

        # 只处理未处理的任务
        unprocessed_tasks = [task for task in self.tasks if task not in existing_tasks]

        # 获取caption
        caption = None
        content_json_path = str(Path(image_path).parent.parent / "content.json")
        if os.path.exists(content_json_path):
            from data_loader import load_captions
            captions = load_captions(content_json_path)
            figure_name = Path(image_path).name
            caption = captions.get("images/" + figure_name, None)

        # 获取各任务的标注字典，字典为空或不存在则对应任务不执行
        def _get_task_dict(key: str) -> Optional[Dict]:
            raw = annotation.get(key) or {}
            return {k: "" for k in raw} if raw else None

        task_inputs = {
            'classification': _get_task_dict('classification'),
            'summarization': _get_task_dict('summarization'),
            'data_extraction': _get_task_dict('data_extraction'),
            'vqa': annotation.get('vqa') or None,
        }

        # 根据字典是否存在，进一步过滤本次需要执行的任务
        # task_input 为空的任务直接写入 {} 占位，不参与推理
        empty_input_tasks = [t for t in unprocessed_tasks if not task_inputs.get(t)]
        unprocessed_tasks = [t for t in unprocessed_tasks if task_inputs.get(t)]

        # 对 task_input 为空的任务，直接保存 {} 结果
        if empty_input_tasks:
            empty_results = {
                task: [{"sample_id": sample_id, task: {}}]
                for task in empty_input_tasks
            }
            await self._append_single_result_async(output_path, empty_results)

        # 运行异步推理（只针对未处理的任务）
        try:
            results = await self.inferencer.run_all_tasks_async(
                image_path,
                caption=caption,
                task_inputs=task_inputs,
                tasks_to_run=unprocessed_tasks,
                pdf_path=pdf_path
            )

            # 推理成功，保存结果（只保存未处理过的任务）
            sample_results = {task: [] for task in unprocessed_tasks}
            for task in unprocessed_tasks:
                if task not in results:
                    continue
                try:
                    results.get(task)[0]
                except:
                    continue
                task_result = {
                    "sample_id": sample_id,
                    task: results.get(task)[0],
                    # "reasoning": results.get(task)[1]
                }
                sample_results[task].append(task_result)

            # 立即追加保存到文件（只保存未处理的任务结果）
            await self._append_single_result_async(output_path, sample_results)

            # 更新进度条
            if pbar:
                pbar.update(1)
                pbar.set_postfix_str(f"Last: {sample_id.split('/')[-1][:20]}")

        except Exception as e:
            traceback.print_exc()
            print(f"    Error during inference: {e}")
            if pbar:
                pbar.write(f"  Error during inference for {sample_id}: {e}")
                pbar.update(1)

    def _get_existing_tasks(self, sample_id: str, output_dir: str, tasks: List[str]) -> List[str]:
        """获取已存在的任务列表（使用缓存快速判断）"""
        existing_tasks = []
        for task in tasks:
            if sample_id in self.all_sample_ids.get(task, set()):
                existing_tasks.append(task)
        return existing_tasks

    def _load_all_sample_ids(self):
        """从已有的结果文件中加载所有任务的 sample_id 集合"""
        output_path = Path(self.output_root)
        self.all_sample_ids = {}

        for task in self.tasks:
            result_file = output_path / f"{task}_results_{self.task_version_tags[task]}.json"
            if result_file.exists():
                with open(result_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                    sample_ids = {r['sample_id'] for r in results}
                    self.all_sample_ids[task] = sample_ids
            else:
                self.all_sample_ids[task] = set()

    async def _filter_unprocessed_samples_async(self, samples: List[Dict[str, Any]],
                                                output_dir: str, tasks: List[str]) -> tuple:
        """
        异步筛选未处理任何任务的样本
        
        Args:
            samples: 所有样本列表
            output_dir: 输出目录
            tasks: 任务列表
            
        Returns:
            (未处理任何任务的样本列表, 已完全处理数量)
        """
        import aiofiles

        output_path = Path(output_dir)
        unprocessed_samples = []
        processed_count = 0

        # 先加载所有结果文件的sample_id集合
        all_sample_ids = {}
        for task in tasks:
            result_file = output_path / f"{task}_results_{self.task_version_tags[task]}.json"
            if result_file.exists():
                async with aiofiles.open(result_file, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    results = json.loads(content)
                    sample_ids = {r['sample_id'] for r in results}
                    all_sample_ids[task] = sample_ids
            else:
                all_sample_ids[task] = set()

        # 检查每个样本是否在所有任务的结果中（即完全处理）
        check_tasks = []
        for sample in samples:
            sample_id = sample['sample_id']
            # 创建检查任务
            check_tasks.append(self._check_sample_processed_async(sample_id, all_sample_ids, tasks))

        # 并发执行所有检查
        is_processed_list = await asyncio.gather(*check_tasks)

        # 筛选未完全处理的样本（至少有一个任务未处理）
        for idx, sample in enumerate(samples):
            if not is_processed_list[idx]:
                unprocessed_samples.append(sample)
            else:
                processed_count += 1

        return unprocessed_samples, processed_count

    async def _check_sample_processed_async(self, sample_id: str,
                                            all_sample_ids: Dict[str, set],
                                            tasks: List[str]) -> bool:
        """
        异步检查单个样本是否已完全处理（所有任务都有结果）
        
        Args:
            sample_id: 样本ID
            all_sample_ids: 所有任务的sample_id集合
            tasks: 任务列表
            
        Returns:
            是否已完全处理(在所有任务的结果中)
        """
        for task in tasks:
            if sample_id not in all_sample_ids.get(task, set()):
                return False
        return True

    async def _append_single_result_async(self, output_path: Path, sample_results: Dict[str, List]):
        """
        异步追加单个样本结果到文件（使用异步锁保护防止并发冲突）
        
        Args:
            output_path: 输出目录路径
            sample_results: 单个样本的结果字典 {task: [sample_result]}
        """
        import aiofiles

        for task, results in sample_results.items():
            if not results:
                continue

            result_file = output_path / f"{task}_results_{self.task_version_tags[task]}.json"

            # 使用异步锁保护文件读写，防止并发冲突
            async with self.file_locks[task]:
                # 读取现有结果（使用异步IO）
                existing_results = []
                if result_file.exists():
                    async with aiofiles.open(result_file, 'r', encoding='utf-8') as f:
                        content = await f.read()
                        if content.strip():
                            existing_results = json.loads(content)

                # 追加新结果
                existing_results.extend(results)

                # 写回文件（使用异步IO）
                async with aiofiles.open(result_file, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(existing_results, ensure_ascii=False, indent=2))

                # 更新缓存中的 sample_id 集合
                for result in results:
                    sample_id = result.get('sample_id')
                    if sample_id:
                        self.all_sample_ids[task].add(sample_id)


async def main():
    """主函数 - 异步执行"""
    # 创建流水线
    pipeline = SciImageMinerPipeline()

    # 处理数据
    print(f"Processing {pipeline.split} dataset...")
    output_files = await pipeline.process_data_set()

    print(f"\n{'=' * 60}")
    print("All results saved:")
    for task, file_path in output_files.items():
        print(f"  {task}: {file_path}")
    print(f"{'=' * 60}")

    # 保存最终结果到 submission 目录并打包
    print("\nPreparing submission files...")

    for task, file_path in output_files.items():
        # 创建任务对应目录
        task_dir = Path(f"{config['output']['root']}/submission/{task}")
        task_dir.mkdir(parents=True, exist_ok=True)

        # 创建 ZIP 文件
        zip_filepath = task_dir / "prediction_data.json.zip"
        print(f"  Creating ZIP for {task}: {zip_filepath}")

        with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(file_path, "prediction_data.json")


if __name__ == '__main__':
    # 运行异步主函数
    asyncio.run(main())