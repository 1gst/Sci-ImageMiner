"""
Sci-ImageMiner 评估指标计算模块
提供各种任务所需的评估指标计算函数
"""
import os

import numpy as np
from typing import List, Dict, Any, Union, Tuple
from collections import defaultdict

from numpy import floating
from rouge import Rouge
from bert_score import score as bert_score
import re
os.environ["TRANSFORMERS_OFFLINE"] = "1"

def calculate_accuracy(predictions: List[str], references: List[str]) -> float:
    """
    计算准确率
    
    Args:
        predictions: 预测结果列表
        references: 真实标签列表
        
    Returns:
        准确率 (0-1)
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length")

    if len(predictions) == 0:
        return 0.0

    correct = sum(1 for pred, ref in zip(predictions, references) if pred.lower().strip() == ref.lower().strip())
    accuracy = correct / len(predictions)
    return accuracy


def calculate_precision_recall_f1(predictions: List[str], references: List[str],
                                  average: str = 'macro') -> Tuple[float, float, float]:
    """
    计算精确率、召回率和F1分数
    
    Args:
        predictions: 预测结果列表
        references: 真实标签列表
        average: 平均方式 ('micro', 'macro', 'weighted')
        
    Returns:
        (precision, recall, f1)
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length")

    if len(predictions) == 0:
        return 0.0, 0.0, 0.0

    # 获取所有类别
    all_labels = set(predictions + references)

    if average == 'micro':
        # 微平均：所有类别的总 TP, FP, FN
        tp = fp = fn = 0
        for pred, ref in zip(predictions, references):
            if pred == ref:
                if pred in all_labels:
                    tp += 1
            else:
                if pred in all_labels:
                    fp += 1
                if ref in all_labels:
                    fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    elif average == 'macro':
        # 宏平均：对每个类别计算指标后求平均
        precisions = []
        recalls = []
        f1s = []

        for label in all_labels:
            pred_binary = [1 if p == label else 0 for p in predictions]
            ref_binary = [1 if r == label else 0 for r in references]

            tp = sum(1 for pb, rb in zip(pred_binary, ref_binary) if pb == 1 and rb == 1)
            fp = sum(1 for pb, rb in zip(pred_binary, ref_binary) if pb == 1 and rb == 0)
            fn = sum(1 for pb, rb in zip(pred_binary, ref_binary) if pb == 0 and rb == 1)

            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

            precisions.append(p)
            recalls.append(r)
            f1s.append(f)

        precision = np.mean(precisions) if precisions else 0.0
        recall = np.mean(recalls) if recalls else 0.0
        f1 = np.mean(f1s) if f1s else 0.0

    elif average == 'weighted':
        # 加权平均：按各类别在真实标签中的出现频率加权
        label_counts = defaultdict(int)
        for ref in references:
            label_counts[ref] += 1

        precisions = []
        recalls = []
        f1s = []

        for label in all_labels:
            pred_binary = [1 if p == label else 0 for p in predictions]
            ref_binary = [1 if r == label else 0 for r in references]

            tp = sum(1 for pb, rb in zip(pred_binary, ref_binary) if pb == 1 and rb == 1)
            fp = sum(1 for pb, rb in zip(pred_binary, ref_binary) if pb == 1 and rb == 0)
            fn = sum(1 for pb, rb in zip(pred_binary, ref_binary) if pb == 0 and rb == 1)

            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

            precisions.append(p)
            recalls.append(r)
            f1s.append(f)

        total = sum(label_counts.values())
        precision = sum(
            p * label_counts[label] / total for p, label in zip(precisions, all_labels)) if total > 0 else 0.0
        recall = sum(r * label_counts[label] / total for r, label in zip(recalls, all_labels)) if total > 0 else 0.0
        f1 = sum(f * label_counts[label] / total for f, label in zip(f1s, all_labels)) if total > 0 else 0.0

    else:
        raise ValueError(f"Unknown average method: {average}")

    return precision, recall, f1


def calculate_rouge(predictions: List[str], references: List[str],
                    rouge_type: str = 'l') -> Dict:
    """
    计算ROUGE分数
    
    Args:
        predictions: 预测结果列表
        references: 真实标签列表
        rouge_type: ROUGE类型 ('1', '2', 'l')
        
    Returns:
        包含ROUGE分数的字典 (precision, recall, fmeasure)
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length")

    if len(predictions) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}

    rouge = Rouge()

    scores = []
    for pred, ref in zip(predictions, references):
        if not pred or not ref:
            scores.append({'p': 0.0, 'r': 0.0, 'f': 0.0})
            continue

        try:
            score = rouge.get_scores(pred, ref, avg=True)
            key = f'rouge-{rouge_type}'
            if key in score:
                scores.append(score[key])
            else:
                scores.append({'p': 0.0, 'r': 0.0, 'f': 0.0})
        except:
            scores.append({'p': 0.0, 'r': 0.0, 'f': 0.0})

    # 计算平均分
    avg_precision = np.mean([s['p'] for s in scores])
    avg_recall = np.mean([s['r'] for s in scores])
    avg_fmeasure = np.mean([s['f'] for s in scores])

    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'fmeasure': avg_fmeasure
    }


def calculate_bertscore(predictions: List[str], references: List[str],
                        model_type: str = '/mnt/8T-1/lingzhou/Resource/ModelResource/google-bert/bert-base-uncased',
                        device: str = 'cpu') -> Tuple[float, float, float]:
    """
    计算BERTScore
    
    Args:
        predictions: 预测结果列表
        references: 真实标签列表
        model_type: BERT模型类型
        device: 计算设备 ('cpu' 或 'cuda')
        
    Returns:
        (precision, recall, f1)
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length")

    if len(predictions) == 0:
        return 0.0, 0.0, 0.0

    try:
        P, R, F1 = bert_score(
            predictions,
            references,
            model_type=model_type,
            num_layers=9,
            device=device,
            verbose=False
        )

        precision = P.mean().item()
        recall = R.mean().item()
        f1 = F1.mean().item()

        return precision, recall, f1
    except Exception as e:
        print(f"Warning: BERTScore calculation failed: {e}")
        return 0.0, 0.0, 0.0


def calculate_exact_match(predictions: List[str], references: List[str]) -> float:
    """
    计算精确匹配率
    
    Args:
        predictions: 预测结果列表
        references: 真实标签列表
        
    Returns:
        精确匹配率 (0-1)
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length")

    if len(predictions) == 0:
        return 0.0

    correct = 0
    for pred, ref in zip(predictions, references):
        if pred.lower().strip() == ref.lower().strip():
            correct += 1

    em = correct / len(predictions)
    return em


def calculate_set_f1(predictions: List[List[str]], references: List[List[str]],
                     order_insensitive: bool = True) -> tuple:
    """
    计算基于集合的Precision, Recall和F1分数 (用于List类型答案)
    
    Args:
        predictions: 预测结果列表 (每个元素是一个字符串列表)
        references: 真实标签列表 (每个元素是一个字符串列表)
        order_insensitive: 是否忽略顺序
        
    Returns:
        (precision, recall, f1) 三个指标 (0-1)
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length")

    if len(predictions) == 0:
        return 0.0, 0.0, 0.0

    precisions = []
    recalls = []
    f1s = []

    for pred_list, ref_list in zip(predictions, references):
        # 转换为小写并去除空白
        pred_set = set(p.lower().strip() for p in pred_list if p.strip())
        ref_set = set(r.lower().strip() for r in ref_list if r.strip())

        if not ref_set and not pred_set:
            precisions.append(1.0)
            recalls.append(1.0)
            f1s.append(1.0)
            continue

        if not ref_set:
            precisions.append(0.0)
            recalls.append(1.0)
            f1s.append(0.0)
            continue

        if not pred_set:
            precisions.append(0.0)
            recalls.append(0.0)
            f1s.append(0.0)
            continue

        # 计算交集
        intersection = pred_set & ref_set

        precision = len(intersection) / len(pred_set)
        recall = len(intersection) / len(ref_set)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return np.mean(precisions), np.mean(recalls), np.mean(f1s)


def calculate_teds(predictions: List[str], references: List[str]) -> float:
    """
    计算TEDS (Tree Edit Distance Similarity) - 简化版本
    
    注意: 完整的TEDS需要解析HTML表格并计算树编辑距离,
    这里提供一个基于字符串相似度的近似实现
    
    Args:
        predictions: 预测结果列表 (markdown表格字符串)
        references: 真实标签列表 (markdown表格字符串)
        
    Returns:
        平均TEDS分数 (0-1)
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length")

    if len(predictions) == 0:
        return 0.0

    def extract_cells(table_str: str) -> List[str]:
        """从markdown表格中提取所有单元格内容"""
        cells = []
        lines = table_str.strip().split('\n')
        for line in lines:
            if '|' in line and not line.strip().startswith('|-'):
                # 移除首尾的 |
                line = line.strip()
                if line.startswith('|'):
                    line = line[1:]
                if line.endswith('|'):
                    line = line[:-1]
                # 分割并清理单元格
                cells_in_line = [cell.strip() for cell in line.split('|')]
                cells.extend(cells_in_line)
        return cells

    def calculate_similarity(cells1: List[str], cells2: List[str]) -> float:
        """计算两个单元格列表的相似度"""
        if not cells1 and not cells2:
            return 1.0

        if not cells1 or not cells2:
            return 0.0

        set1 = set(c.lower() for c in cells1 if c)
        set2 = set(c.lower() for c in cells2 if c)

        intersection = set1 & set2
        union = set1 | set2

        return len(intersection) / len(union) if union else 0.0

    teds_scores = []
    for pred, ref in zip(predictions, references):
        pred_cells = extract_cells(pred)
        ref_cells = extract_cells(ref)

        similarity = calculate_similarity(pred_cells, ref_cells)
        teds_scores.append(similarity)

    return np.mean(teds_scores)


def calculate_rms(predictions: List[str], references: List[str]) -> float:
    """
    计算RMS (Relative Mapping Similarity) - 简化版本
    
    注意: 完整的RMS需要建立单元格之间的映射关系,
    这里提供一个基于关键词覆盖率的近似实现
    
    Args:
        predictions: 预测结果列表 (markdown表格字符串)
        references: 真实标签列表 (markdown表格字符串)
        
    Returns:
        平均RMS分数 (0-1)
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length")

    if len(predictions) == 0:
        return 0.0

    def extract_keywords(table_str: str) -> set:
        """从表格中提取关键词(统计词频)"""
        # 移除markdown标记
        table_str = re.sub(r'\|\s*-+\s*\|', '', table_str)  # 移除分隔线
        table_str = table_str.replace('|', ' ')  # 替换管道符

        # 分词
        words = re.findall(r'\b\w+\b', table_str.lower())

        # 过滤停用词和数字
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                      'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                      'should', 'may', 'might', 'must', 'shall'}

        keywords = set()
        for word in words:
            if word not in stop_words and not word.isdigit():
                keywords.add(word)

        return keywords

    def calculate_mapping_coverage(pred_keywords: set, ref_keywords: set) -> float:
        """计算关键词映射覆盖率"""
        if not ref_keywords:
            return 1.0 if not pred_keywords else 0.0

        if not pred_keywords:
            return 0.0

        # 计算预测关键词对真实关键词的覆盖
        coverage = len(pred_keywords & ref_keywords) / len(ref_keywords)

        return coverage

    rms_scores = []
    for pred, ref in zip(predictions, references):
        pred_keywords = extract_keywords(pred)
        ref_keywords = extract_keywords(ref)

        coverage = calculate_mapping_coverage(pred_keywords, ref_keywords)
        rms_scores.append(coverage)

    return np.mean(rms_scores)
