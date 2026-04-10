"""
Sci-ImageMiner 竞赛任务推理器
使用大模型完成四个任务：分类、数据提取、摘要和VQA
"""
import ast
import os
import json
import re

import fitz
import yaml
import base64
from typing import Dict, List, Any, Optional
from pathlib import Path
from infer_utils import OpenAIInfer
from utils import encode_image, build_messages, pdf_pages_to_base64

# 加载各任务提示词配置文件
def _load_task_prompts(task: str, version: str) -> dict:
    """
    加载指定任务和版本的提示词。

    Args:
        task: 任务名称，支持 "classification" / "data_extraction" / "summarization" / "vqa"
        version: 版本号字符串，如 "v1", "v2", "v3", "v4"

    Returns:
        对应版本的提示词字典（含 system / user / example_reference_template 等字段）

    Raises:
        FileNotFoundError: 提示词配置文件不存在
        KeyError: 指定版本在配置文件中不存在
    """
    config_path = os.path.join(os.path.dirname(__file__), f"../config/{task}_prompts.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        all_versions = yaml.safe_load(f)
    if version not in all_versions:
        available = list(all_versions.keys())
        raise KeyError(f"提示词版本 '{version}' 在 {task}_prompts.yaml 中不存在，可用版本: {available}")
    return all_versions[version]


def _load_vqa_instructions(version: str) -> dict:
    """
    加载 VQA 答案类型说明。

    Args:
        version: 版本号字符串，如 "v1", "v2"

    Returns:
        答案类型说明字典
    """
    config_path = os.path.join(os.path.dirname(__file__), "../config/vqa_prompts.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        all_versions = yaml.safe_load(f)
    key = f"instructions_{version}"
    if key not in all_versions:
        available = [k for k in all_versions if k.startswith("instructions_")]
        raise KeyError(f"VQA 答案类型说明版本 '{key}' 不存在，可用: {available}")
    return all_versions[key]


with open("../config/figure_taxonomy_v4.yaml", "r", encoding="utf-8") as f:
    CLASS = yaml.safe_load(f)

# ImageTypeMap PDF 示例图片路径
EXAMPLE_PDF_PATH = os.path.join(os.path.dirname(__file__), "../data/ImageTypeMapV2.pdf")


class SciFigureInference:
    """科学图表推理器"""

    def __init__(self, model_name: str, base_url: str, api_key: str,
                 proxy_url: Optional[str] = None, max_workers: int = 3,
                 config_tasks: Optional[List[str]] = None,
                 prompt_versions: Optional[Dict[str, str]] = None):
        """
        初始化推理器

        Args:
            model_name: 模型名称，如 "qwen3-vl-8b-instruct"
            base_url: API基础URL
            api_key: API密钥
            proxy_url: 代理URL（可选）
            max_workers: 最大并发数
            config_tasks: 配置文件中指定的任务列表（可选），如 ["classification", "data_extraction", "summarization", "vqa"]
                        如果为None，则默认为所有4个任务
            prompt_versions: 各任务使用的提示词版本（可选），格式如：
                        {
                            "classification": "v4",
                            "data_extraction": "v2",
                            "summarization": "v2",
                            "vqa": "v2"
                        }
                        未指定的任务将使用默认版本（各任务最新版本）。
        """
        self.model_name = model_name
        self.client = OpenAIInfer(
            base_url=base_url,
            api_key=api_key,
            proxy_url=proxy_url
        )
        self.max_workers = max_workers
        self.classification_classes = CLASS
        self.model_args = {
            "extra_body": {
                'enable_thinking': True
            }
        }
        # 图片编码缓存
        self._image_cache = {}

        # 解析各任务提示词版本，未指定时使用各任务默认（最新）版本
        _default_versions = {
            "classification": "v4",
            "data_extraction": "v2",
            "summarization": "v2",
            "vqa": "v2",
        }
        _versions = {**_default_versions, **(prompt_versions or {})}

        # 加载各任务提示词
        self.classification_prompt = _load_task_prompts("classification", _versions["classification"])
        self.data_extract_prompt = _load_task_prompts("data_extraction", _versions["data_extraction"])
        self.summarize_prompt = _load_task_prompts("summarization", _versions["summarization"])
        self.vqa_prompt = _load_task_prompts("vqa", _versions["vqa"])
        self._vqa_instructions = _load_vqa_instructions(_versions["vqa"])

        # 各任务版本标签字典，供外部按任务拼接文件名，格式: {"classification": "v4", ...}
        self.task_version_tags = dict(_versions)

        print(f"提示词版本: classification={_versions['classification']}, "
              f"data_extraction={_versions['data_extraction']}, "
              f"summarization={_versions['summarization']}, "
              f"vqa={_versions['vqa']}")

        # 加载 ImageTypeMap 示例图片（PDF转base64）
        self._example_images = []
        self._example_reference_text = ""
        if os.path.exists(EXAMPLE_PDF_PATH):
            try:
                self._example_images = pdf_pages_to_base64(EXAMPLE_PDF_PATH)
                # 使用提示词模板生成示例引用文本
                example_template = self.classification_prompt.get("example_reference_template", "")
                if example_template:
                    self._example_reference_text = example_template
            except Exception as e:
                print(f"Warning: Failed to load ImageTypeMap examples: {e}")
                self._example_images = []
                self._example_reference_text = ""

        # 配置的任务列表（如果未指定，默认为所有4个任务）
        self.config_tasks = config_tasks if config_tasks else ["classification", "data_extraction", "summarization",
                                                               "vqa"]

    def _prepare_message(self, prompt: str, image_path: str, system_prompt: str, pdf_path: Optional[str] = None) -> \
            List[Dict[str, Any]]:
        """准备包含图片的消息格式"""
        # 使用全文一起推理效果略微下降，暂时不用
        # pdf_path = None
        # 检查缓存，编码图片
        if image_path in self._image_cache:
            base64_image = self._image_cache[image_path]
        else:
            # 编码并缓存
            base64_image = encode_image(image_path)
            self._image_cache[image_path] = base64_image
        # 编码PDF页面
        pdf_pages = []
        if pdf_path:
            if pdf_path in self._image_cache:
                pdf_pages = self._image_cache[pdf_path]
            else:
                pdf_pages = pdf_pages_to_base64(pdf_path)
                self._image_cache[pdf_path] = pdf_pages
        base64_images = [base64_image] + pdf_pages
        return build_messages(prompt, base64_images, system_prompt)

    async def classify_figure(self, image_path: str, caption: Optional[str] = None,
                              uncategorized_dict: Dict = None, pdf_path: str = None) -> tuple[dict[str, Any], Any]:
        """
        任务1:图表分类 - 异步版本

        Args:
            image_path: 图片路径
            caption: 可选的图片标题
            uncategorized_dict: 未分类字典，格式:{"a": "类别名", "b": "类别名", ...}
            pdf_path: 可选的PDF路径

        Returns:
            分类结果字典，格式:{"a": "类别名", "b": "类别名", ...}
        """
        # 构建提示词（填充示例引用文本）
        prompt = self.classification_prompt["user"].format(
            class_count=len(self.classification_classes),
            classes=self.classification_classes,
            uncategorized_dict=uncategorized_dict if uncategorized_dict else {},
            caption=caption if caption else "",
            example_reference=self._example_reference_text
        )

        # 构建消息
        messages = self._prepare_message(prompt, image_path, self.classification_prompt["system"], pdf_path)

        # 将示例图片附加到消息中（作为额外的图片内容）
        if self._example_images:
            user_msg = messages[-1]  # 最后一条消息是user消息
            for example_img in self._example_images:
                user_msg["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{example_img}"
                    }
                })

        # 异步调用大模型
        content, reasoning = await self.client.ainvoke(self.model_name, messages, **self.model_args)

        # 解析结果
        try:
            result = self._parse_json_response(content)
        except:
            raise ValueError("分类json解析失败")
            print("分类json解析失败")
            return None
        return result, reasoning

    async def extract_data_table(self, image_path: str, caption: Optional[str] = None,
                                 data_extraction_dict: Optional[Dict[str, Any]] = None,
                                 pdf_path: str = None) -> tuple[
        dict[str, Any], Any]:
        """
        任务2:数据表提取 - 异步版本

        Args:
            image_path: 图片路径
            caption: 可选的图片标题
            data_extraction_dict: 数据提取任务的子图字典（可选），格式 {"a": "", ...}
            pdf_path: 可选的PDF路径

        Returns:
            提取的数据表字典，格式:{"a": "markdown表格", "b": "markdown表格", ...}
            如果图表类型无法提取数据（如聚类图、SEM/TEM图像等），则返回空字符串 ""
        """
        # 构建提示词
        prompt = self.data_extract_prompt["user"].format(
            caption=caption if caption else ""
        )

        # 构建消息
        messages = self._prepare_message(prompt, image_path, self.data_extract_prompt["system"], pdf_path)

        # 异步调用大模型
        content, reasoning = await self.client.ainvoke(self.model_name, messages, **self.model_args)
        # 解析结果
        try:
            result = self._parse_json_response(content)
            # 替换markdown标记，json标记应该保留
            for key, value in result.items():
                pattern = r'```markdown\n(.*?)\n```'
                result[key] = re.sub(pattern, r'\1', value, flags=re.DOTALL).replace("```", "")
        except:
            raise ValueError("表格json解析失败")
            print("表格json解析失败")
            return None

        return result, reasoning

    async def summarize_figure(self, image_path: str, caption: Optional[str] = None,
                               summarization_dict: Optional[Dict[str, Any]] = None,
                               pdf_path: str = None) -> tuple[
        dict[str, Any], Any]:
        """
        任务3:图表摘要 - 异步版本

        Args:
            image_path: 图片路径
            caption: 可选的图片标题
            summarization_dict: 摘要任务的子图字典（可选），格式 {"a": "", ...}
            pdf_path: 可选的PDF路径

        Returns:
            摘要字典，格式:{"a": "摘要文本", "b": "摘要文本", ...}
        """
        # 构建提示词
        prompt = self.summarize_prompt["user"].format(
            caption=caption if caption else ""
        )

        # 构建消息
        messages = self._prepare_message(prompt, image_path, self.summarize_prompt["system"], pdf_path)

        # 异步调用大模型
        content, reasoning = await self.client.ainvoke(self.model_name, messages, **self.model_args)

        # 解析结果
        try:
            result = self._parse_json_response(content)
        except:
            raise ValueError("总结json解析失败")
            print("总结json解析失败")
            return None

        return result, reasoning

    async def generate_vqa(self, image_path: str, caption: Optional[str] = None,
                           questions: Optional[List[Dict[str, str]]] = None, pdf_path: str = None) -> tuple[
        dict[str, Any], Any]:
        """
        任务4:视觉问答 (VQA) - 异步版本（优化：并发处理多个问题）

        Args:
            image_path: 图片路径
            caption: 可选的图片标题
            questions: 可选的问题列表，每个问题包含 question_type, question, answer_type
                      如果为None，则返回空字典
            pdf_path: 可选的PDF路径

        Returns:
            VQA结果字典，格式:{"a": [{"question_type": "...", "question": "...", "answer_type": "...", "answer": "..."}], ...}
            如果没有提供问题，返回空字典 {}
        """

        if questions:
            # 如果提供了问题，则并发回答这些问题
            vqa_results = {}

            # 创建所有问题的并发任务
            qa_tasks = []
            task_keys = []

            for key, qa_list in questions.items():
                for q_info in qa_list:
                    question = f"Regarding Subfigure {key}:" + q_info['question']
                    # 创建异步任务
                    qa_task = self._answer_single_question(
                        image_path,
                        question,
                        q_info['answer_type'],
                        caption,
                        pdf_path
                    )
                    qa_tasks.append(qa_task)
                    task_keys.append((key, q_info))

            # 并发执行所有问答任务
            answers = await asyncio.gather(*qa_tasks)

            # 组织结果
            for (key, q_info), answer in zip(task_keys, answers):
                if key not in vqa_results:
                    vqa_results[key] = []
                vqa_results[key].append({
                    'question_type': q_info['question_type'],
                    'question': q_info['question'],
                    'answer_type': q_info['answer_type'],
                    'answer': answer[0]
                    # 'reasoning': answer[1]
                })

            return vqa_results, ""
        else:
            # 如果没有提供问题，返回空字典
            return {}, ""

    async def _answer_single_question(self, image_path: str, question: str,
                                      answer_type: str, caption: Optional[str] = None, pdf_path: str = None) -> tuple[
        str, str]:
        """
        回答单个问题 - 异步版本

        Args:
            image_path: 图片路径
            question: 问题文本
            answer_type: 答案类型 (Yes/No, Factoid, List, Paragraph)
            caption: 可选的标题
            pdf_path: 可选的PDF路径

        Returns:
            答案文本
        """
        # 获取答案类型说明（使用实例加载的版本）
        instruction = self._vqa_instructions.get(
            answer_type,
            "Provide a clear and accurate answer."
        )

        # 构建提示词
        prompt = self.vqa_prompt["user"].format(
            question=question,
            answer_type=answer_type,
            instruction=instruction,
            caption=caption if caption else ""
        )

        # 构建消息
        messages = self._prepare_message(prompt, image_path, self.vqa_prompt["system"], pdf_path)

        # 异步调用大模型
        content, reasoning = await self.client.ainvoke(self.model_name, messages, **self.model_args)

        # 清理答案
        answer = content.strip()

        # 根据答案类型进行后处理
        if answer_type == "Yes/No":
            if "yes" in answer.lower():
                answer = "Yes"
            elif "no" in answer.lower():
                answer = "No"

        return answer, reasoning

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        解析JSON格式的响应（已优化，更健壮）

        处理步骤:
        1. 提取被 ```json ... ``` 包裹的内容。
        2. 如果没有代码块，则寻找最外层的 `{...}` 来提取主体。
        3. 尝试用 `json.loads()` 标准解析。
        4. 如果失败（例如因为单引号），则尝试用 `ast.literal_eval()` 解析。
        5. 如果全部失败，抛出异常。

        Args:
            response: 大模型返回的文本

        Returns:
            解析后的字典
        """
        response = response.strip()

        # 步骤 1: 提取 JSON 内容的字符串
        json_str = response

        # 优先处理 ```json 代码块
        if "```json" in response:
            try:
                json_start = response.find("```json") + 7
                json_end = response.rfind("```")
                if json_end > json_start:
                    json_str = response[json_start:json_end].strip()
            except Exception:
                # 如果提取出错，则退回到使用原始 response
                pass
        # 如果没有代码块，尝试找到最外层的 { 和 }
        else:
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx: end_idx + 1]

        # 步骤 2: 尝试用标准 json 库解析
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # 步骤 3: 如果标准库失败，尝试用 ast.literal_eval 处理单引号等情况
            try:
                return ast.literal_eval(json_str)
            except (ValueError, SyntaxError, MemoryError) as e:
                # 如果所有方法都失败了，抛出一个清晰的错误
                raise ValueError(
                    f"无法将响应解析为JSON或Python字典。\n"
                    f"原始响应: '{response}'\n"
                    f"尝试解析的部分: '{json_str}'"
                ) from e

    async def run_all_tasks_async(self, image_path: str, caption: Optional[str] = None,
                                  task_inputs: Optional[Dict[str, Any]] = None,
                                  tasks_to_run: Optional[List[str]] = None, pdf_path: str = None) -> Dict[str, Any]:
        """
        异步运行指定任务列表

        Args:
            image_path: 图片路径
            caption: 可选的标题
            task_inputs: 各任务的标注数据字典，key 为任务名，value 为对应的标注内容：
                         {
                             "classification": {"a": "", ...},  # 待分类子图字典
                             "data_extraction": {"a": "", ...}, # 待提取子图字典
                             "summarization":   {"a": "", ...}, # 待摘要子图字典
                             "vqa":             {"a": [...], ...} # VQA问题字典
                         }
                         某任务的 value 为 None 或缺失时，该任务不会被执行。
            tasks_to_run: 要运行的任务列表，如 ["classification", "data_extraction"]。
                         如果为None，则运行所有配置的任务（self.config_tasks）。
            pdf_path: pdf路径

        Returns:
            包含所有配置任务的结果字典，包括未执行的任务（返回空字典）
        """
        results = {}

        # 使用配置的任务列表作为基准
        all_tasks = self.config_tasks

        # 如果 tasks_to_run 为 None，则运行所有配置的任务
        if tasks_to_run is None:
            tasks_to_run = all_tasks

        # 过滤掉不在配置中的任务
        tasks_to_run = [task for task in tasks_to_run if task in all_tasks]

        # 为所有配置的任务初始化结果字典（包括未执行的任务）
        for task in all_tasks:
            results[task] = {}, ""

        # 从 task_inputs 中解包各任务数据
        _inputs = task_inputs or {}
        uncategorized_dict = _inputs.get("classification")
        data_extraction_dict = _inputs.get("data_extraction")
        summarization_dict = _inputs.get("summarization")
        vqa_questions = _inputs.get("vqa")

        # 只处理指定的未完成任务
        tasks_list = []

        if "classification" in tasks_to_run:
            tasks_list.append(
                ("classification", self.classify_figure(image_path, caption, uncategorized_dict, pdf_path)))

        if "data_extraction" in tasks_to_run:
            tasks_list.append(("data_extraction", self.extract_data_table(image_path, caption, data_extraction_dict, pdf_path)))

        if "summarization" in tasks_to_run:
            tasks_list.append(("summarization", self.summarize_figure(image_path, caption, summarization_dict, pdf_path)))

        if "vqa" in tasks_to_run and vqa_questions:
            tasks_list.append(("vqa", self.generate_vqa(image_path, caption, vqa_questions, pdf_path)))

        # 并发执行所有任务
        task_results = await asyncio.gather(*[task[1] for task in tasks_list])

        # 提取结果
        for idx, (task_name, _) in enumerate(tasks_list):
            if task_results[idx] is not None:
                results[task_name] = task_results[idx]

        return results


# 导入asyncio
import asyncio

if __name__ == '__main__':
    # 测试推理器
    async def test_inferencer():
        # 设置API密钥（从环境变量中读取）
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        api_key = os.environ.get('ALIYUN_API_KEY', '')
        model_name = "qwen3-vl-8b-instruct"

        if not api_key:
            print("Please set ALIYUN_API_KEY environment variable")
            return

        # 创建推理器（示例：指定各任务提示词版本）
        inferencer = SciFigureInference(
            model_name, base_url, api_key, max_workers=3,
            prompt_versions={
                "classification": "v4",
                "data_extraction": "v2",
                "summarization": "v2",
                "vqa": "v2",
            }
        )

        # 测试单张图片
        test_image = "icdar2026-competition-data/dev/atomic-layer-deposition/experimental-usecase/11/images/fig1.jpg"

        if os.path.exists(test_image):
            print("Running inference on test image...")
            results = await inferencer.run_all_tasks_async(test_image)
            print("\nResults:")
            print(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            print(f"Test image not found: {test_image}")


    # 运行测试
    asyncio.run(test_inferencer())
