#!/usr/bin/env python3
"""
Sci-ImageMiner 评估主程序
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluator import UnifiedEvaluator


def main():
    parser = argparse.ArgumentParser(description='Sci-ImageMiner评估程序')
    parser.add_argument('--config', type=str, default='../../config/config.yaml', help='配置文件路径')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda', help='计算设备')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Sci-ImageMiner 评估程序")
    print("=" * 60)
    
    evaluator = UnifiedEvaluator(args.config)
    results = evaluator.evaluate_all(device=args.device)
    
    print("\n" + "=" * 60)
    print("评估完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
