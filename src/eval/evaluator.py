"""
Sci-ImageMiner 统一评估器
整合所有任务的评估流程
"""

import json
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict

from metrics import (
    calculate_accuracy, calculate_precision_recall_f1,
    calculate_rouge, calculate_bertscore,
    calculate_exact_match, calculate_set_f1,
    calculate_rms, calculate_teds
)
from data_loader import SciFigureDataLoader


class UnifiedEvaluator:
    """统一评估器 - 整合所有任务的评估"""

    def __init__(self, config_path: str = "../../config/config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            self.config = config['evaluation'] if 'evaluation' in config else {}

        self.tasks = self.config.get('tasks', [])
        self.data_root = self.config.get('data', {}).get('root', '')
        self.split = self.config.get('data', {}).get('split', 'dev')
        self.results_dir = Path(self.config.get('results_dir', '../../predictions/results'))
        self.output_dir = Path(self.config.get('output_dir', './results'))
        self.thresholds = self.config.get('thresholds', {})

        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"评估器初始化完成")
        print(f"  数据集: {self.split}")
        print(f"  结果目录: {self.results_dir}")
        print(f"  输出目录: {self.output_dir}")

    def load_ground_truth(self) -> Dict[str, Any]:
        """加载ground truth数据"""
        print(f"\n加载ground truth数据...")
        data_loader = SciFigureDataLoader(self.data_root)
        samples = data_loader.load_data_from_split(self.split)

        gt_data = {}
        for sample in samples:
            sample_id = sample['sample_id']
            gt_data[sample_id] = {
                'classification': sample['annotation'].get('classification', {}),
                'data_extraction': sample['annotation'].get('data_extraction', {}),
                'summarization': sample['annotation'].get('summarization', {}),
                'vqa': sample['annotation'].get('vqa', {})
            }

        print(f"加载了 {len(gt_data)} 个样本")
        return gt_data

    def load_predictions(self, task_name: str) -> List[Dict[str, Any]]:
        """加载指定任务的预测结果"""
        result_file = self.results_dir / f"{task_name}_results.json"

        if not result_file.exists():
            print(f"警告: 预测结果文件不存在: {result_file}")
            return []

        with open(result_file, 'r', encoding='utf-8') as f:
            predictions = json.load(f)

        print(f"加载了 {len(predictions)} 个{task_name}预测结果")
        return predictions

    def _extract_pairs(self, predictions: List[Dict], gt_data: Dict, task_key: str,
                       check_not_empty: bool = False) -> Tuple[List, List, List]:
        """提取预测和ground truth对
        
        Args:
            predictions: 预测结果列表
            gt_data: ground truth数据
            task_key: 任务键名 (如 'classification', 'summarization')
            check_not_empty: 是否检查值非空
            
        Returns:
            (preds, refs, sample_ids) 预测列表、参考列表、样本ID列表
        """
        preds, refs, sample_ids = [], [], []

        for pred_item in predictions:
            sample_id = pred_item['sample_id']
            if sample_id not in gt_data:
                continue

            sample_ids.append(sample_id)
            pred_task = pred_item.get(task_key, {})
            ref_task = gt_data[sample_id].get(task_key, {})

            for key in ref_task:
                if key in pred_task:
                    pred_val = pred_task[key]
                    ref_val = ref_task[key]

                    if check_not_empty and (not pred_val or not ref_val):
                        continue

                    preds.append(pred_val)
                    refs.append(ref_val)

        return preds, refs, sample_ids

    def evaluate_all(self, device: str = 'cpu') -> Dict[str, Any]:
        """评估所有任务"""
        gt_data = self.load_ground_truth()
        all_results = {}

        # 分类任务
        if 'classification' in self.tasks:
            all_results['classification'] = self._evaluate_classification(gt_data)

        # 数据抽取任务
        if 'data_extraction' in self.tasks:
            all_results['data_extraction'] = self._evaluate_data_extraction(gt_data)

        # 总结任务
        if 'summarization' in self.tasks:
            all_results['summarization'] = self._evaluate_summarization(gt_data, device)

        # VQA任务
        if 'vqa' in self.tasks:
            all_results['vqa'] = self._evaluate_vqa(gt_data, device)

        # 生成汇总报告
        self._save_summary(all_results)

        return all_results

    def _evaluate_classification(self, gt_data: Dict[str, Any]) -> Dict[str, Any]:
        """评估分类任务"""
        print("\n" + "=" * 50)
        print("评估分类任务")
        print("=" * 50)

        predictions = self.load_predictions('classification')
        if not predictions:
            return {}

        all_preds, all_refs, sample_ids = self._extract_pairs(predictions, gt_data, 'classification')

        # 计算指标
        accuracy = calculate_accuracy(all_preds, all_refs)
        precision, recall, f1 = calculate_precision_recall_f1(all_preds, all_refs)

        results = {
            'overall': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'total_count': len(all_preds)
            },
            'sample_ids': sample_ids
        }

        # 识别低分样本
        f1_threshold = self.thresholds.get('classification', {}).get('f1_threshold', 0.8)
        low_samples = self._find_low_classification(predictions, gt_data, f1_threshold)
        results['low_score_samples'] = low_samples

        self._print_results('分类', results['overall'])
        self._save_results('classification', results)

        return results

    def _find_low_classification(self, predictions: List, gt_data: Dict, threshold: float) -> List:
        """识别分类低分样本"""
        low_samples = []
        for pred_item in predictions:
            sample_id = pred_item['sample_id']
            if sample_id not in gt_data:
                continue

            pred_cls = pred_item.get('classification', {})
            ref_cls = gt_data[sample_id].get('classification', {})

            for key in ref_cls:
                if key in pred_cls and pred_cls[key] != ref_cls[key]:
                    sample = {"sample_id": sample_id, "prediction": pred_cls, "ground_truth": ref_cls}
                    low_samples.append(sample)
                    break

        return low_samples

    def _evaluate_data_extraction(self, gt_data: Dict[str, Any]) -> Dict[str, Any]:
        """评估数据抽取任务"""
        print("\n" + "=" * 50)
        print("评估数据抽取任务")
        print("=" * 50)

        predictions = self.load_predictions('data_extraction')
        if not predictions:
            return {}

        all_preds, all_refs, sample_ids = self._extract_pairs(predictions, gt_data, 'data_extraction',
                                                              check_not_empty=True)

        # 计算指标
        rms = calculate_rms(all_preds, all_refs)
        teds = calculate_teds(all_preds, all_refs)

        results = {
            'overall': {
                'rms': rms,
                'teds': teds,
                'total_count': len(all_preds)
            },
            'sample_ids': sample_ids
        }

        # 识别低分样本
        teds_threshold = self.thresholds.get('data_extraction', {}).get('teds_threshold', 0.7)
        low_samples = self._find_low_data_extraction(predictions, gt_data, teds_threshold)
        results['low_score_samples'] = low_samples

        self._print_results('数据抽取', results['overall'])
        self._save_results('data_extraction', results)

        return results

    def _find_low_data_extraction(self, predictions: List, gt_data: Dict, threshold: float) -> List:
        """识别数据抽取低分样本"""
        low_samples = []
        for pred_item in predictions:
            sample_id = pred_item['sample_id']
            if sample_id not in gt_data:
                continue

            pred_ext = pred_item.get('data_extraction', {})
            ref_ext = gt_data[sample_id].get('data_extraction', {})

            for key in pred_ext:
                if key in ref_ext and pred_ext[key] and ref_ext[key]:
                    teds = calculate_teds([pred_ext[key]], [ref_ext[key]])
                    if teds < threshold:
                        sample = {"sample_id": sample_id, "prediction": pred_ext, "ground_truth": ref_ext}
                        low_samples.append(sample)
                        break

        return low_samples

    def _evaluate_summarization(self, gt_data: Dict[str, Any], device: str) -> Dict[str, Any]:
        """评估总结任务"""
        print("\n" + "=" * 50)
        print("评估总结任务")
        print("=" * 50)

        predictions = self.load_predictions('summarization')
        if not predictions:
            return {}

        all_preds, all_refs, sample_ids = self._extract_pairs(predictions, gt_data, 'summarization',
                                                              check_not_empty=True)

        # 计算ROUGE
        rouge_1 = calculate_rouge(all_preds, all_refs, '1')
        rouge_2 = calculate_rouge(all_preds, all_refs, '2')
        rouge_l = calculate_rouge(all_preds, all_refs, 'l')

        # 计算BERTScore
        bert_p, bert_r, bert_f1 = calculate_bertscore(all_preds, all_refs, device=device)

        results = {
            'overall': {
                'rouge_1': rouge_1,
                'rouge_2': rouge_2,
                'rouge_l': rouge_l,
                'bert_score': {'precision': bert_p, 'recall': bert_r, 'f1': bert_f1},
                'total_count': len(all_preds)
            },
            'sample_ids': sample_ids
        }

        # 识别低分样本
        rouge_threshold = self.thresholds.get('summarization', {}).get('rouge_l_threshold', 0.6)
        low_samples = self._find_low_summarization(predictions, gt_data, rouge_threshold, device)
        results['low_score_samples'] = low_samples

        self._print_results('总结', results['overall'])
        self._save_results('summarization', results)

        return results

    def _find_low_summarization(self, predictions: List, gt_data: Dict, threshold: float, device: str) -> List:
        """识别总结低分样本"""
        low_samples = []
        for pred_item in predictions:
            sample_id = pred_item['sample_id']
            if sample_id not in gt_data:
                continue

            pred_sum = pred_item.get('summarization', {})
            ref_sum = gt_data[sample_id].get('summarization', {})

            for key in ref_sum:
                if key in pred_sum:
                    rouge = calculate_rouge([pred_sum[key]], [ref_sum[key]], 'l')
                    if rouge['fmeasure'] < threshold:
                        sample = {"sample_id": sample_id, "prediction": pred_sum, "ground_truth": ref_sum}
                        low_samples.append(sample)
                        break

        return low_samples

    def _evaluate_vqa(self, gt_data: Dict[str, Any], device: str) -> Dict[str, Any]:
        """评估VQA任务"""
        print("\n" + "=" * 50)
        print("评估VQA任务")
        print("=" * 50)

        predictions = self.load_predictions('vqa')
        if not predictions:
            return {}

        # 按答案类型分类
        type_data = defaultdict(lambda: {'preds': [], 'refs': [], 'sample_ids': []})
        sample_ids = []

        for pred_item in predictions:
            sample_id = pred_item['sample_id']
            if sample_id not in gt_data:
                continue

            sample_ids.append(sample_id)
            pred_vqa = pred_item.get('vqa', {})
            ref_vqa = gt_data[sample_id].get('vqa', {})

            for subfig_key in pred_vqa:
                if subfig_key not in ref_vqa:
                    continue

                for pred_qa, ref_qa in zip(pred_vqa[subfig_key], ref_vqa[subfig_key]):
                    answer_type = pred_qa.get('answer_type', '')
                    if not answer_type:
                        continue

                    type_data[answer_type]['preds'].append(pred_qa.get('answer', ''))
                    type_data[answer_type]['refs'].append(ref_qa.get('answer', ''))

        # 计算每种类型的指标
        type_results = {}
        low_samples_by_type = {}

        for answer_type in ['Yes/No', 'Factoid', 'List', 'Paragraph']:
            preds = type_data[answer_type]['preds']
            refs = type_data[answer_type]['refs']

            if not preds:
                type_results[answer_type] = {'count': 0}
                continue

            if answer_type == 'Yes/No':
                acc = calculate_accuracy(preds, refs)
                prec, rec, f1 = calculate_precision_recall_f1(preds, refs)
                type_results[answer_type] = {
                    'count': len(preds),
                    'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1
                }

            elif answer_type == 'Factoid':
                em = calculate_exact_match(preds, refs)
                rouge_l = calculate_rouge(preds, refs, 'l')
                type_results[answer_type] = {
                    'count': len(preds),
                    'exact_match': em, 'rouge_l': rouge_l
                }

            elif answer_type == 'List':
                pred_lists = [self._split_list(p) for p in preds]
                ref_lists = [self._split_list(r) for r in refs]
                prec, rec, f1 = calculate_set_f1(pred_lists, ref_lists)
                type_results[answer_type] = {
                    'count': len(preds),
                    'precision': prec, 'recall': rec, 'f1': f1
                }

            elif answer_type == 'Paragraph':
                rouge_l = calculate_rouge(preds, refs, 'l')
                bert_p, bert_r, bert_f = calculate_bertscore(preds, refs, device=device)
                type_results[answer_type] = {
                    'count': len(preds),
                    'rouge_l': rouge_l,
                    'bert_score': {'precision': bert_p, 'recall': bert_r, 'f1': bert_f}
                }

            # 识别低分样本
            threshold_config = self.thresholds.get('vqa', {}).get(answer_type.lower(), {})
            if threshold_config:
                low_samples = self._find_low_vqa(predictions, gt_data, answer_type, threshold_config, device)
                low_samples_by_type[answer_type] = low_samples

        results = {
            'overall': type_results,
            'low_score_samples': low_samples_by_type,
            'sample_ids': sample_ids
        }

        self._print_vqa_results(type_results)
        self._save_results('vqa', results)

        return results

    def _split_list(self, answer: str) -> List[str]:
        """分割List类型答案"""
        items = [answer]
        for sep in [',', ';', '\n', '•', '-']:
            new_items = []
            for item in items:
                new_items.extend(item.split(sep))
            items = new_items
        return [item.strip() for item in items if item.strip()]

    def _find_low_vqa(self, predictions: List, gt_data: Dict, answer_type: str,
                      thresholds: Dict, device: str) -> List:
        """识别VQA低分样本"""
        low_samples = []
        for pred_item in predictions:
            sample_id = pred_item['sample_id']
            if sample_id not in gt_data:
                continue

            pred_vqa = pred_item.get('vqa', {})
            ref_vqa = gt_data[sample_id].get('vqa', {})

            for subfig_key in pred_vqa:
                if subfig_key not in ref_vqa:
                    continue

                for pred_qa, ref_qa in zip(pred_vqa[subfig_key], ref_vqa[subfig_key]):
                    if pred_qa.get('answer_type') != answer_type:
                        continue

                    pred_answer = pred_qa.get('answer', '')
                    ref_answer = ref_qa.get('answer', '')

                    is_low = False
                    if answer_type == 'Yes/No':
                        _, _, f1 = calculate_precision_recall_f1([pred_answer], [ref_answer])
                        is_low = f1 < thresholds.get('f1_threshold', 0.8)

                    elif answer_type == 'Factoid':
                        em = calculate_exact_match([pred_answer], [ref_answer])
                        is_low = em < thresholds.get('em_threshold', 0.7)

                    elif answer_type == 'List':
                        pred_list, ref_list = self._split_list(pred_answer), self._split_list(ref_answer)
                        _, _, f1 = calculate_set_f1([pred_list], [ref_list])
                        is_low = f1 < thresholds.get('f1_threshold', 0.7)

                    elif answer_type == 'Paragraph':
                        rouge = calculate_rouge([pred_answer], [ref_answer], 'l')
                        is_low = rouge['fmeasure'] < thresholds.get('rouge_l_threshold', 0.6)

                    if is_low:
                        sample = {"sample_id": sample_id, "prediction": pred_answer, "ground_truth": ref_answer}
                        low_samples.append(sample)
                        break

        return low_samples

    def _print_results(self, task_name: str, metrics: Dict):
        """打印评估结果"""
        print(f"\n{task_name}任务指标:")
        for key, value in metrics.items():
            if key == 'total_count':
                print(f"  样本数: {value}")
            elif isinstance(value, dict):
                for k2, v2 in value.items():
                    if isinstance(v2, float):
                        print(f"  {key}/{k2}: {v2:.4f}")
                    else:
                        print(f"  {key}/{k2}: {v2}")
            elif isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    def _print_vqa_results(self, type_results: Dict):
        """打印VQA评估结果"""
        print("\nVQA任务指标:")
        for answer_type, metrics in type_results.items():
            if metrics.get('count', 0) == 0:
                continue
            print(f"\n  {answer_type} (n={metrics['count']}):")
            for key, value in metrics.items():
                if key != 'count':
                    if isinstance(value, dict):
                        for k2, v2 in value.items():
                            if isinstance(v2, float):
                                print(f"    {key}/{k2}: {v2:.4f}")
                    elif isinstance(value, float):
                        print(f"    {key}: {value:.4f}")

    def _save_results(self, task_name: str, results: Dict):
        """保存评估结果"""
        # JSON格式
        json_file = self.output_dir / f"{task_name}_results_eval.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 低分样本列表
        if 'low_score_samples' in results:
            low_file = self.output_dir / f"{task_name}_low_samples.json"
            low_data = {
                'task': task_name,
                'low_count': len(results['low_score_samples']),
                'low_samples': results['low_score_samples']
            }
            with open(low_file, 'w', encoding='utf-8') as f:
                json.dump(low_data, f, ensure_ascii=False, indent=2)

        print(f"结果已保存到: {json_file}")

    def _save_summary(self, all_results: Dict):
        """保存汇总报告"""
        report_file = self.output_dir / f"summary_report.txt"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("评估结果汇总\n")
            f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")

            for task_name, results in all_results.items():
                if not results:
                    continue

                f.write(f"{task_name.upper()}\n")
                f.write("-" * 40 + "\n")

                metrics = results.get('overall', {})
                for key, value in metrics.items():
                    if key == 'total_count':
                        f.write(f"  样本数: {value}\n")
                    elif isinstance(value, dict):
                        for k2, v2 in value.items():
                            if isinstance(v2, float):
                                f.write(f"  {key}/{k2}: {v2:.4f}\n")
                    elif isinstance(value, float):
                        f.write(f"  {key}: {value:.4f}\n")

                low_count = len(results.get('low_score_samples', {}))
                if isinstance(low_count, int):
                    f.write(f"  低分样本数: {low_count}\n")

                f.write("\n")

            f.write("=" * 60 + "\n")

        print(f"\n汇总报告已保存到: {report_file}")
