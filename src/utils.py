import base64
from pathlib import Path
from typing import List, Dict, Any

import fitz


def encode_image(image_path: str) -> str:
    """将图片编码为base64"""
    # 编码并缓存
    with open(image_path, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')
        return encoded


def pdf_pages_to_base64(pdf_path: str, dpi: int = 200, fmt: str = "png") -> List[Dict[str, Any]]:
    """
    使用 PyMuPDF 将PDF的每一页转换为base64编码格式。
    无需安装 poppler。
    """
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

    # 打开PDF文件
    doc = fitz.open(pdf_path)

    if doc.page_count == 0:
        raise ValueError(f"PDF文件为空: {pdf_path}")

    result = []
    # 计算缩放因子 (dpi / 72，因为PDF默认72dpi)
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        # 将页面渲染为图片
        pix = page.get_pixmap(matrix=matrix)

        # 转换为指定格式
        if fmt.lower() == "jpeg":
            image_bytes = pix.tobytes(output="jpeg")
            mime_type = "image/jpeg"
        elif fmt.lower() == "png":
            image_bytes = pix.tobytes(output="png")
            mime_type = "image/png"
        else:
            image_bytes = pix.tobytes(output="jpeg")
            mime_type = "image/jpeg"

        # 编码为base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        result.append(base64_image)
        # print(f"已处理第 {page_num + 1} 页，共 {doc.page_count} 页")
    doc.close()
    return result


def build_messages(user_prompt: str, base64images: List[str], system_prompt: str) -> List[Dict[str, Any]]:
    """
    构建消息列表

    Args:
        user_prompt: 用户提示词
        base64images: 图片编码集合
        system_prompt: 系统提示词

    Returns:
        构建好的消息列表
    """
    messages = []
    # 准备系统提示词信息
    if system_prompt:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        })
    # 准备用户消息内容
    user_contents = [{"type": "text", "text": user_prompt}]
    for base64image in base64images:
        user_contents.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64image}"
            }
        })
    # 构建消息列表
    messages.append({
        "role": "user",
        "content": user_contents
    })
    return messages
