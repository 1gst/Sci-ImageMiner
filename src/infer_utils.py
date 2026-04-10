import os
import random
import traceback
from http import HTTPStatus

import httpx
from openai import AsyncOpenAI, OpenAI
import time
import dashscope
import google.genai as genai
from google.genai import types
from google.genai.types import GenerateContentConfig, ThinkingConfig
import asyncio
import base64
import magic
from typing import Optional, Tuple, List
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
    retry_if_exception_type,
    AsyncRetrying,
    Retrying
)


class OpenAIInfer:

    def __init__(self, base_url, api_key, extra_body=None, proxy_url="socks5://127.0.0.1:13659", is_retry=False):
        self.api_key = api_key
        self.async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=600,
            http_client=httpx.AsyncClient(proxy=proxy_url) if "api.openai.com" in base_url else httpx.AsyncClient(
                verify=False)
        )

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=600,
            http_client=httpx.Client(proxy=proxy_url) if "api.openai.com" in base_url else httpx.Client(verify=False)
        )
        self.gemini_client = GeminiInfer()
        self.extra_body = extra_body
        # 设置重试配置
        if is_retry:
            self.max_retries = 3
            self.retry_wait = 300  # 300秒
        else:
            self.max_retries = 1
            self.retry_wait = 1  # 300秒

    async def ainvoke(self, model, messages, **args):
        # 使用 AsyncRetrying 包装逻辑
        # 它可以捕获异常并自动等待重试，而不需要写 while 循环
        try:
            async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(self.max_retries),
                    wait=wait_fixed(self.retry_wait),
                    # 指定捕获的异常，建议捕获 Exception 或具体的 APIError
                    retry=retry_if_exception_type(Exception),
                    reraise=True  # 最后一次失败后抛出异常
            ):
                with attempt:
                    return await self._ainvoke_core(model, messages, **args)
        except Exception as e:
            print(f"[{model}] 最终调用失败: {e}")
            return None, None

    async def _ainvoke_core(self, model, messages, **args):
        if 'llama' in model:
            return self.llama_invoke(model, messages)
        if 'gemini' in model:
            return await self.gemini_client.ainvoke(model, messages)

        if "extra_body" in args:
            self.extra_body = args.pop("extra_body", None)

        chat_response = await self.async_client.chat.completions.create(
            model=model,
            messages=messages if isinstance(messages, list) else [{"role": "user", "content": messages}],
            extra_body=self.extra_body if self.extra_body else None,
            **args,
            # temperature=0.7,
            # top_p=0.8,
            # presence_penalty=1.5,
            # extra_body={
            #     "top_k": 20,
            #     "enable_search": enable_search
            # }
        )
        content = chat_response.choices[0].message.content
        # Assume reasoning_content might not be present in async mode
        if content and "</think>" in content:
            reasoning_content = content.split("</think>")[0].split("<think>")[1]
            content = content.split("</think>")[1]
            if "<answer>" in content:
                content = content.split("<answer>")[1]
            if "</answer>" in content:
                content = content.split("</answer>")[0]
        else:
            reasoning_content = chat_response.choices[0].message.model_extra.get('reasoning_content', '')
        return content, reasoning_content

    def invoke(self, model, messages, **args):
        """同步方法的重试逻辑"""
        try:
            for attempt in Retrying(
                    stop=stop_after_attempt(self.max_retries),
                    wait=wait_fixed(self.retry_wait),
                    retry=retry_if_exception_type(Exception),
                    reraise=True
            ):
                with attempt:
                    return self._invoke_core(model, messages, **args)
        except Exception as e:
            print(f"[{model}] 同步调用最终失败: {e}")
            return None, None

    def _invoke_core(self, model, messages, **args):
        if 'llama' in model:
            return self.llama_invoke(model, messages)
        if 'gemini' in model:
            return self.gemini_client.invoke(model, messages)

        if "extra_body" in args:
            self.extra_body = args.pop("extra_body", None)
        chat_response = self.client.chat.completions.create(
            model=model,
            messages=messages if isinstance(messages, list) else [{"role": "user", "content": messages}],
            extra_body=self.extra_body if self.extra_body else None,
            **args
            # temperature=0.3,
            # top_p=0.8,
            # presence_penalty=1.5,
            # extra_body={
            #     "top_k": 20,
            #     "enable_search": enable_search
            # },
        )
        content = chat_response.choices[0].message.content
        # Assume reasoning_content might not be present in async mode
        if content and "</think>" in content:
            reasoning_content = content.split("</think>")[0].split("<think>")[1]
            content = content.split("</think>")[1]
            if "<answer>" in content:
                content = content.split("<answer>")[1]
            if "</answer>" in content:
                content = content.split("</answer>")[0]
        else:
            reasoning_content = chat_response.choices[0].message.model_extra.get('reasoning_content', '')
        return content, reasoning_content

    def llama_invoke(self, model, messages, **args):
        if "extra_body" in args:
            self.extra_body = args.pop("extra_body", None)
        chat_response = dashscope.MultiModalConversation.call(
            # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx",
            api_key=self.api_key,
            model=model,
            messages=messages,
            extra_body=self.extra_body if self.extra_body else None,
            **args
        )
        if chat_response.status_code == HTTPStatus.OK:
            content = chat_response.output.choices[0].message.content[0]["text"]
            if content and "</think>" in content:
                reasoning_content = content.split("</think>")[0].split("<think>")[1]
                content = content.split("</think>")[1]
                if "<answer>" in content:
                    content = content.split("<answer>")[1]
                if "</answer>" in content:
                    content = content.split("</answer>")[0]
            else:
                reasoning_content = chat_response.choices[0].message.model_extra.get('reasoning_content', '')
            return content, reasoning_content
        else:
            print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                chat_response.request_id, chat_response.status_code,
                chat_response.code, chat_response.message
            ))

    async def astream(self, model, messages, **args):
        if "extra_body" in args:
            self.extra_body = args.pop("extra_body", None)
        completion = await self.async_client.chat.completions.create(
            model=model,
            messages=messages if isinstance(messages, list) else [{"role": "user", "content": messages}],
            extra_body=self.extra_body if self.extra_body else None,
            stream=True,
            **args
        )

        reasoning_content = ""
        answer_content = ""
        is_answering = False

        async for chunk in completion:  # 使用 async for 迭代异步流
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                reasoning_content += delta.reasoning_content

            if hasattr(delta, "content") and delta.content:
                if not is_answering:
                    is_answering = True
                answer_content += delta.content

        return answer_content, reasoning_content

    def stream(self, model, messages, **args):
        if "extra_body" in args:
            self.extra_body = args.pop("extra_body", None)
        print(self.extra_body)
        completion = self.client.chat.completions.create(
            model=model,
            messages=messages if isinstance(messages, list) else [{"role": "user", "content": messages}],
            extra_body=self.extra_body if self.extra_body else None,
            stream=True,
            **args
        )

        reasoning_content = ""
        answer_content = ""
        is_answering = False

        for chunk in completion:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                reasoning_content += delta.reasoning_content
                print(delta.reasoning_content, end="", flush=True)

            if hasattr(delta, "content") and delta.content:
                if not is_answering:
                    is_answering = True
                answer_content += delta.content
                print(delta.content, end="", flush=True)

        return answer_content, reasoning_content


class GeminiInfer:
    PROJECT_ID = "zoloz-471609"
    REGION = "global"
    KEY_PATH = r"/Users/lingzhou/ProjectConfig/zoloz-471609-25b6ccc38d98.json"

    # KEY_PATH = r"C:\Users\LingZhou\zoloz-471609-dbe0b1adf88.json"

    def __init__(self, key_path: str = KEY_PATH, project_id: str = PROJECT_ID, region: str = REGION):
        """
        初始化 Gemini 客户端

        Args:
            project_id: Google Cloud 项目 ID
            region: 区域
            KEY_PATH: 服务账户密钥
        """
        import os
        os.environ['GOOGLE_CLOUD_PROJECT'] = project_id
        os.environ['GOOGLE_CLOUD_LOCATION'] = region
        os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = 'True'
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_path

        # 创建同步客户端
        self.client = genai.Client(vertexai=True, project=project_id, location=region)

    def _extract_content_from_messages(self, messages: list) -> Tuple[str, List[dict], Optional[str]]:
        """
        从消息列表中提取文本内容和图像数据列表

        Args:
            messages: 消息列表

        Returns:
            Tuple[prompt_text, image_list, mime_type]
            image_list: 包含多个 {'data': bytes, 'mime_type': str} 字典的列表
            mime_type: 主要图片类型（保留以兼容旧逻辑，建议新逻辑使用image_list中的信息）
        """
        prompt_text = ""
        image_list = []  # 改为列表存储多张图片
        primary_mime_type = None  # 重命名以更清晰

        for message in messages:
            role = message["role"]
            content = message["content"]

            # 处理系统消息
            if role == "system":
                prompt_text += f"{content}\n"

            # 处理用户消息
            elif role == "user":
                if isinstance(content, str):
                    prompt_text += f"{content}\n"
                elif isinstance(content, list):
                    for item in content:
                        if item["type"] == "text":
                            prompt_text += f"{item['text']}\n"
                        elif item["type"] == "image_url":
                            # 处理 base64 图像
                            image_url = item["image_url"]["url"]
                            if image_url.startswith("data:image"):
                                # 提取 base64 数据
                                base64_data = image_url.split(",")[1]
                                image_data = base64.b64decode(base64_data)

                                # 自动检测 MIME 类型
                                try:
                                    import magic
                                    detected_mime = magic.from_buffer(image_data, mime=True)
                                except:
                                    # 回退方案：从 data URL 提取 MIME
                                    mime_part = image_url.split(":")[1].split(";")[0]
                                    detected_mime = mime_part if mime_part else "image/jpeg"

                                # 添加到图片列表
                                image_list.append({
                                    'data': image_data,
                                    'mime_type': detected_mime
                                })

                                # 记录第一张图片的 MIME 类型（兼容旧逻辑）
                                if not primary_mime_type:
                                    primary_mime_type = detected_mime

        return prompt_text, image_list, primary_mime_type

    def invoke(self, model: str, messages: list, ) -> Tuple[str, str]:
        """
        同步调用 Gemini 模型（已支持多图）
        """
        try:
            # 从消息中提取文本和图像列表
            prompt_text, image_list, mime_type = self._extract_content_from_messages(messages)

            # 准备内容 - 文本部分
            contents = [prompt_text]

            # 添加所有图片到内容
            for img_info in image_list:
                contents.append(types.Part.from_bytes(
                    data=img_info['data'],
                    mime_type=img_info['mime_type']
                ))

            # 配置生成参数
            config = GenerateContentConfig(
                thinking_config=ThinkingConfig(include_thoughts=True)
            )

            # 生成内容
            response = self.client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )

            # 提取响应内容
            content = response.text.strip()

            # 提取推理内容
            reasoning_content = self._extract_reasoning_content(response)

            return content, reasoning_content

        except Exception as e:
            traceback.print_exc()
            print(f"Gemini 同步调用出错: {e}")
            raise e

    async def ainvoke(self, model: str, messages: list, ) -> Tuple[str, str]:
        """
        异步调用 Gemini 模型（已支持多图）
        """
        try:
            # 从消息中提取文本和图像列表
            prompt_text, image_list, mime_type = self._extract_content_from_messages(messages)

            # 准备内容 - 文本部分
            contents = [prompt_text]

            # 添加所有图片到内容
            for img_info in image_list:
                contents.append(types.Part.from_bytes(
                    data=img_info['data'],
                    mime_type=img_info['mime_type']
                ))

            # 配置生成参数
            config = GenerateContentConfig(
                thinking_config=ThinkingConfig(include_thoughts=True)
            )

            # 使用线程池执行同步操作以避免阻塞
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config
                )
            )

            # 提取响应内容
            content = response.text.strip()

            # 提取推理内容
            reasoning_content = self._extract_reasoning_content(response)

            return content, reasoning_content

        except Exception as e:
            print(f"Gemini 异步调用出错: {e}")
            raise e

    def print_usage(self, usage):
        """
        打印 Gemini 使用情况
        """
        print(f"输入 Token 数 (Prompt): {usage.prompt_token_count}")
        print(f"输出 Token 数 (Candidates): {usage.candidates_token_count}")
        print(f"总 Token 数 (Total): {usage.total_token_count}")

        # 如果开启了思考功能（Gemini 2.5 Pro 特色），还可以查看思考 Token
        if hasattr(usage, 'thoughts_token_count'):
            print(f"思考过程 Token 数: {usage.thoughts_token_count}")

    def _extract_reasoning_content(self, response) -> str:
        """
        从 Gemini 响应中提取推理内容

        Args:
            response: Gemini 响应对象

        Returns:
            推理内容字符串
        """
        reasoning_content = ""

        try:
            # 如果响应中包含思考过程，提取它
            # 注意：这取决于 Gemini 模型是否返回思考内容
            # 根据您的实际响应结构调整此方法
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                # 检查是否有思考内容
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            # 这里可以根据实际响应结构进一步提取推理内容
                            text = part.text
                            # 如果文本中包含类似 <think> 标签的内容，提取它
                            if "</think>" in text:
                                reasoning_content = text.split("</think>")[0].split("<think>")[1]
                                break

        except Exception as e:
            print(f"提取推理内容时出错: {e}")

        return reasoning_content


class DashScopeInfer:

    def __init__(self, api_key):
        self.api_key = api_key

    def search_invoke(self, model, messages, enable_search=True, enable_thinking=False):
        response = dashscope.Generation.call(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
            api_key=self.api_key,
            model=model,  # 此处以qwen-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=messages if isinstance(messages, list) else [{"role": "user", "content": messages}],
            enable_search=enable_search,
            search_options={
                "forced_search": True,  # 强制开启联网搜索
                "enable_source": True,  # 使返回结果包含搜索来源的信息，OpenAI 兼容方式暂不支持返回
            },
            incremental_output=True if enable_thinking else False,
            stream=True if enable_thinking else False,
            enable_thinking=enable_thinking,
            result_format='message'
        )
        if enable_thinking:
            # 定义完整思考过程
            reasoning_content = ""
            # 定义完整回复
            answer_content = ""
            # 判断是否结束思考过程并开始回复
            is_answering = False
            # 判断是否为第一个chunk，便于打印搜索信息
            is_first_chunk = True
            search_results = None
            for chunk in response:
                if is_first_chunk:
                    search_results = chunk.output.search_info["search_results"]
                    reasoning_content += chunk.output.choices[0].message.reasoning_content
                    is_first_chunk = False
                else:
                    # 如果思考过程与回复皆为空，则忽略
                    if (chunk.output.choices[0].message.content == "" and
                            chunk.output.choices[0].message.reasoning_content == ""):
                        pass
                    else:
                        # 如果当前为思考过程
                        if (chunk.output.choices[0].message.reasoning_content != "" and
                                chunk.output.choices[0].message.content == ""):
                            reasoning_content += chunk.output.choices[0].message.reasoning_content
                        # 如果当前为回复
                        elif chunk.output.choices[0].message.content != "":
                            if not is_answering:
                                is_answering = True
                            answer_content += chunk.output.choices[0].message.content
            content = "<think>\n" + reasoning_content + "</think>\n" + answer_content
        else:
            content = response.output.choices[0].message.content
            search_results = response.output.search_info["search_results"]
        return content, search_results


if __name__ == '__main__':
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key = os.environ.get('ALIYUN_API_KEY')
    llm_client = OpenAIInfer(base_url, api_key)

    openai_client = OpenAIInfer(r"https://api.openai.com/v1", os.environ.get("OPENAI_API_KEY"))

    # 初始化 Gemini 客户端
    gemini_client = GeminiInfer()
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "你是一个有用的助手"}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请你介绍一下你自己，是否支持多模态的输入？"},
            ],
        }
    ]
    content, reasoning = llm_client.invoke("qwen3-vl-8b-instruct", messages)

    content, reasoning = openai_client.invoke("gpt-5", messages)
    # 同步调用
    content, reasoning = gemini_client.invoke("gemini-2.5-pro", messages)

    # 异步调用
    content, reasoning = asyncio.run(gemini_client.ainvoke("gemini-3-preview", messages))

    llm_client = OpenAIInfer("http://0.0.0.0:8001/v1", "EMPTY")
    while True:
        content, reasoning = llm_client.invoke("qwen3-vl-8b-111903", messages)
        print(content)
        time.sleep(600)
