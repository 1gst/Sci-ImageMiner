r"""
从 prediction_data 中筛选指定 answer_type 的问题，调用大模型重新回答，生成新的结果文件。

提示词和推理逻辑与 main.py / task_inferencer.py 中 VQA 任务一致。

用法示例：
    # 重新回答所有 answer_type 为 Factoid 的问题
    python update_vqa_results.py \
        --input "predictions/prediction_data 1.json" \
        --output predictions/prediction_data_updated.json \
        --answer-type Factoid

    # 重新回答多种 answer_type
    python update_vqa_results.py \
        --input "predictions/prediction_data 1.json" \
        --output predictions/prediction_data_updated.json \
        --answer-type Factoid Paragraph

    # 指定模型、提示词版本、数据集根目录等
    python update_vqa_results.py \
        --input "predictions/prediction_data 1.json" \
        --output predictions/prediction_data_updated.json \
        --answer-type Factoid \
        --model qwen3.6-plus \
        --prompt-version v4 \
        --data-root /path/to/icdar2026-competition-data \
        --split dev \
        --max-workers 16
"""

import argparse
import asyncio
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml
from tqdm import tqdm

# 将当前脚本所在目录加入 sys.path，以便导入同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import SciFigureDataLoader, load_captions
from infer_utils import OpenAIInfer
from utils import encode_image, build_messages, pdf_pages_to_base64


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------

def load_config() -> dict:
    """加载 config.yaml 中的 inference 配置"""
    config_path = os.path.join(os.path.dirname(__file__), "../config/config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["inference"]


def load_vqa_prompts(version: str) -> dict:
    """加载 VQA 提示词"""
    config_path = os.path.join(os.path.dirname(__file__), "../config/vqa_prompts.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        all_versions = yaml.safe_load(f)
    if version not in all_versions:
        available = [k for k in all_versions if not k.startswith("instructions_")]
        raise KeyError(f"VQA 提示词版本 '{version}' 不存在，可用: {available}")
    return all_versions[version]


def load_vqa_instructions(version: str) -> dict:
    """加载 VQA 答案类型说明"""
    config_path = os.path.join(os.path.dirname(__file__), "../config/vqa_prompts.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        all_versions = yaml.safe_load(f)
    key = f"instructions_{version}"
    if key not in all_versions:
        available = [k for k in all_versions if k.startswith("instructions_")]
        raise KeyError(f"VQA 答案类型说明版本 '{key}' 不存在，可用: {available}")
    return all_versions[key]


# ---------------------------------------------------------------------------
# 数据集索引构建
# ---------------------------------------------------------------------------

def build_dataset_index(data_root: str, split: str) -> Dict[str, Dict[str, Any]]:
    """
    加载数据集并构建 sample_id -> sample 的索引。

    Returns:
        {sample_id: {"image_path": ..., "pdf_path": ..., "caption": ...}}
    """
    loader = SciFigureDataLoader(data_root)
    samples = loader.load_data_from_split(split)

    index: Dict[str, Dict[str, Any]] = {}
    for sample in samples:
        sample_id = sample["sample_id"]
        image_path = sample["image_path"]
        pdf_path = sample.get("pdf_path")

        # 加载 caption
        caption = None
        content_json_path = str(Path(image_path).parent.parent / "content.json")
        if os.path.exists(content_json_path):
            captions = load_captions(content_json_path)
            figure_name = Path(image_path).name
            caption = captions.get("images/" + figure_name, None)

        index[sample_id] = {
            "image_path": image_path,
            "pdf_path": pdf_path,
            "caption": caption,
        }
    return index


# ---------------------------------------------------------------------------
# 单问题推理
# ---------------------------------------------------------------------------

class VQAReInferencer:
    """VQA 重推理器，复用 task_inferencer.py 中的提示词和推理逻辑"""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        proxy_url: Optional[str] = None,
        prompt_version: str = "v4",
    ):
        self.model_name = model_name
        self.client = OpenAIInfer(
            base_url=base_url,
            api_key=api_key,
            proxy_url=proxy_url,
        )
        self.model_args = {"extra_body": {"enable_thinking": True}}

        # 加载提示词
        self.vqa_prompt = load_vqa_prompts(prompt_version)
        self.vqa_instructions = load_vqa_instructions(prompt_version)

        # 图片编码缓存
        self._image_cache: Dict[str, Any] = {}

    def _prepare_message(
        self, prompt: str, image_path: str, system_prompt: str, pdf_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """准备包含图片的消息格式（与 task_inferencer.py 一致）"""
        # 编码图片（带缓存）
        if image_path in self._image_cache:
            base64_image = self._image_cache[image_path]
        else:
            base64_image = encode_image(image_path)
            self._image_cache[image_path] = base64_image

        # 编码 PDF 页面（带缓存）
        pdf_pages: list = []
        if pdf_path:
            if pdf_path in self._image_cache:
                pdf_pages = self._image_cache[pdf_path]
            else:
                pdf_pages = pdf_pages_to_base64(pdf_path)
                self._image_cache[pdf_path] = pdf_pages

        base64_images = [base64_image] + pdf_pages
        return build_messages(prompt, base64_images, system_prompt)

    async def answer_question(
        self,
        image_path: str,
        subfig_key: str,
        question: str,
        answer_type: str,
        caption: Optional[str] = None,
        pdf_path: Optional[str] = None,
    ) -> str:
        """
        回答单个 VQA 问题（与 task_inferencer._answer_single_question 逻辑一致）
        """
        # 获取答案类型说明
        instruction = self.vqa_instructions.get(
            answer_type, "Provide a clear and accurate answer."
        )

        # 构建提示词（与 task_inferencer.generate_vqa 中拼接方式一致）
        full_question = f"Regarding Subfigure {subfig_key}:" + question
        prompt = self.vqa_prompt["user"].format(
            question=full_question,
            answer_type=answer_type,
            instruction=instruction,
            caption=caption if caption else "",
        )

        # 构建消息
        messages = self._prepare_message(
            prompt, image_path, self.vqa_prompt["system"], pdf_path
        )

        # 调用大模型
        content, reasoning = await self.client.ainvoke(
            self.model_name, messages, **self.model_args
        )

        if content is None:
            return ""

        answer = content.strip()

        # Yes/No 后处理
        if answer_type == "Yes/No":
            if "yes" in answer.lower():
                answer = "Yes"
            elif "no" in answer.lower():
                answer = "No"

        return answer


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def collect_questions_to_reinfer(
    data: list, target_types: Set[str]
) -> List[Dict[str, Any]]:
    """
    从 prediction_data 中收集需要重新推理的问题。

    Returns:
        列表，每个元素: {
            "sample_idx": int,          # 在 data 中的索引
            "sample_id": str,
            "subfig_key": str,
            "question_idx": int,        # 在 questions 列表中的索引
            "question": str,
            "question_type": str,
            "answer_type": str,
        }
    """
    target_lower = {t.lower() for t in target_types}
    tasks = []
    for sample_idx, sample in enumerate(data):
        sample_id = sample["sample_id"]
        vqa = sample.get("vqa", {})
        if not vqa:
            continue
        for subfig_key, questions in vqa.items():
            for q_idx, q_item in enumerate(questions):
                if q_item.get("answer_type", "").lower() in target_lower:
                    tasks.append({
                        "sample_idx": sample_idx,
                        "sample_id": sample_id,
                        "subfig_key": subfig_key,
                        "question_idx": q_idx,
                        "question": q_item["question"],
                        "question_type": q_item["question_type"],
                        "answer_type": q_item["answer_type"],
                    })
    return tasks


async def reinfer_and_update(
    data: list,
    tasks: List[Dict[str, Any]],
    dataset_index: Dict[str, Dict[str, Any]],
    inferencer: VQAReInferencer,
    max_workers: int = 16,
) -> Tuple[list, dict]:
    """
    并发重新推理并更新 data 中的答案。

    Returns:
        (updated_data, stats)
    """
    updated = copy.deepcopy(data)
    stats = {"total": len(tasks), "success": 0, "failed": 0, "skipped_no_data": 0}

    semaphore = asyncio.Semaphore(max_workers)

    async def _process_one(task_info: dict, pbar: tqdm) -> None:
        sample_id = task_info["sample_id"]
        ds_info = dataset_index.get(sample_id)
        if not ds_info:
            stats["skipped_no_data"] += 1
            pbar.update(1)
            return

        async with semaphore:
            try:
                answer = await inferencer.answer_question(
                    image_path=ds_info["image_path"],
                    subfig_key=task_info["subfig_key"],
                    question=task_info["question"],
                    answer_type=task_info["answer_type"],
                    caption=ds_info.get("caption"),
                    pdf_path=ds_info.get("pdf_path"),
                )
                # 写回结果
                sample = updated[task_info["sample_idx"]]
                q_list = sample["vqa"][task_info["subfig_key"]]
                q_list[task_info["question_idx"]]["answer"] = answer
                stats["success"] += 1
            except Exception as e:
                stats["failed"] += 1
                pbar.write(f"  Error [{sample_id}]: {e}")
            finally:
                pbar.update(1)

    with tqdm(total=len(tasks), desc="Re-inferring VQA", unit="q") as pbar:
        coros = [_process_one(t, pbar) for t in tasks]
        await asyncio.gather(*coros)

    return updated, stats


async def async_main(args: argparse.Namespace) -> None:
    """异步主函数"""
    # 加载配置（作为默认值）
    config = load_config()

    # 参数优先，否则使用配置文件的值
    model_name = args.model or config["model"]["name"]
    base_url = args.base_url or config["model"]["base_url"]
    proxy_url = args.proxy_url if args.proxy_url is not None else config["model"].get("proxy_url")
    prompt_version = args.prompt_version or config.get("prompt_versions", {}).get("vqa", "v4")
    data_root = args.data_root or config["data"]["root"]
    split = args.split or config["data"]["split"]
    max_workers = args.max_workers or config["concurrency"]["max_workers"]

    # 解析 API Key
    api_key_template = config["model"]["api_key"]
    if api_key_template.startswith("${") and api_key_template.endswith("}"):
        env_var = api_key_template[2:-1]
        api_key = os.environ.get(env_var, "")
        if not api_key:
            raise ValueError(f"API key not found in environment variable: {env_var}")
    else:
        api_key = api_key_template

    # 1. 加载输入文件
    print(f"加载输入文件: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  -> {len(data)} 个样本")

    target_types = set(args.answer_type)
    print(f"目标 answer_type: {target_types}")

    # 2. 收集需要重新推理的问题
    tasks = collect_questions_to_reinfer(data, target_types)
    print(f"需要重新推理的问题数: {len(tasks)}")
    if not tasks:
        print("没有需要重新推理的问题，直接复制输入文件到输出。")
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return

    # 3. 构建数据集索引（sample_id -> 图片/PDF/caption 路径）
    print(f"加载数据集索引: {data_root}/{split} ...")
    dataset_index = build_dataset_index(data_root, split)
    print(f"  -> 数据集中 {len(dataset_index)} 个样本")

    # 4. 初始化推理器
    print(f"模型: {model_name}, 提示词版本: {prompt_version}, 并发数: {max_workers}")
    inferencer = VQAReInferencer(
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        proxy_url=proxy_url,
        prompt_version=prompt_version,
    )

    # 5. 并发重新推理
    updated_data, stats = await reinfer_and_update(
        data, tasks, dataset_index, inferencer, max_workers
    )

    # 6. 保存结果
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(updated_data, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {args.output}")

    # 7. 输出统计
    print("\n===== 重推理统计 =====")
    print(f"总问题数:          {stats['total']}")
    print(f"成功:              {stats['success']}")
    print(f"失败:              {stats['failed']}")
    print(f"数据集中未找到:    {stats['skipped_no_data']}")


def main():
    parser = argparse.ArgumentParser(
        description="从 prediction_data 中筛选指定 answer_type 的问题，调用大模型重新回答"
    )
    parser.add_argument(
        "--input", required=True,
        help="输入的 prediction_data JSON 文件路径",
    )
    parser.add_argument(
        "--output", required=True,
        help="输出文件路径",
    )
    parser.add_argument(
        "--answer-type", nargs="+", required=True,
        help="需要重新回答的 answer_type（支持多个，不区分大小写），例如: Factoid Paragraph",
    )
    parser.add_argument(
        "--model", default=None,
        help="模型名称（默认使用 config.yaml 中的配置）",
    )
    parser.add_argument(
        "--base-url", default=None,
        help="API base URL（默认使用 config.yaml 中的配置）",
    )
    parser.add_argument(
        "--proxy-url", default=None,
        help="代理 URL（默认使用 config.yaml 中的配置）",
    )
    parser.add_argument(
        "--prompt-version", default=None,
        help="VQA 提示词版本，如 v2, v3, v4（默认使用 config.yaml 中的配置）",
    )
    parser.add_argument(
        "--data-root", default=None,
        help="数据集根目录（默认使用 config.yaml 中的配置）",
    )
    parser.add_argument(
        "--split", default=None,
        help="数据集分割，如 dev, test（默认使用 config.yaml 中的配置）",
    )
    parser.add_argument(
        "--max-workers", type=int, default=None,
        help="最大并发数（默认使用 config.yaml 中的配置）",
    )

    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()

