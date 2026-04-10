import json
import csv
import os

GEMINI_PATH = "results_dev_gemini-3.1-pro_v3/classification_low_samples.json"
QWEN_PATH = "results_dev_qwen3.5-plus_type_pdf_v3/classification_low_samples.json"
OUTPUT_PATH = "low_samples_comparison.csv"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_json(relative_path):
    full_path = os.path.join(SCRIPT_DIR, relative_path)
    with open(full_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_sample_map(samples):
    return {sample["sample_id"]: sample for sample in samples}


def main():
    gemini_data = load_json(GEMINI_PATH)
    qwen_data = load_json(QWEN_PATH)

    gemini_map = build_sample_map(gemini_data["low_samples"])
    qwen_map = build_sample_map(qwen_data["low_samples"])

    all_sample_ids = sorted(set(gemini_map.keys()) | set(qwen_map.keys()))

    rows = []

    for sample_id in all_sample_ids:
        in_gemini = sample_id in gemini_map
        in_qwen = sample_id in qwen_map

        if in_gemini and in_qwen:
            gemini_pred = gemini_map[sample_id]["prediction"]
            qwen_pred = qwen_map[sample_id]["prediction"]
            ground_truth = gemini_map[sample_id]["ground_truth"]

            if gemini_pred != qwen_pred:
                rows.append({
                    "sample_id": sample_id,
                    "diff_type": "prediction_mismatch",
                    "gemini_prediction": json.dumps(gemini_pred, ensure_ascii=False),
                    "qwen_prediction": json.dumps(qwen_pred, ensure_ascii=False),
                    "ground_truth": json.dumps(ground_truth, ensure_ascii=False),
                })

        elif in_gemini and not in_qwen:
            gemini_pred = gemini_map[sample_id]["prediction"]
            ground_truth = gemini_map[sample_id]["ground_truth"]
            rows.append({
                "sample_id": sample_id,
                "diff_type": "only_in_gemini",
                "gemini_prediction": json.dumps(gemini_pred, ensure_ascii=False),
                "qwen_prediction": "",
                "ground_truth": json.dumps(ground_truth, ensure_ascii=False),
            })

        elif not in_gemini and in_qwen:
            qwen_pred = qwen_map[sample_id]["prediction"]
            ground_truth = qwen_map[sample_id]["ground_truth"]
            rows.append({
                "sample_id": sample_id,
                "diff_type": "only_in_qwen",
                "gemini_prediction": "",
                "qwen_prediction": json.dumps(qwen_pred, ensure_ascii=False),
                "ground_truth": json.dumps(ground_truth, ensure_ascii=False),
            })

    output_path = os.path.join(SCRIPT_DIR, OUTPUT_PATH)
    fieldnames = ["sample_id", "diff_type", "gemini_prediction", "qwen_prediction", "ground_truth"]

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # 打印统计摘要
    mismatch_count = sum(1 for r in rows if r["diff_type"] == "prediction_mismatch")
    only_gemini_count = sum(1 for r in rows if r["diff_type"] == "only_in_gemini")
    only_qwen_count = sum(1 for r in rows if r["diff_type"] == "only_in_qwen")

    print(f"=== 比较结果 ===")
    print(f"Gemini low_samples 总数: {len(gemini_map)}")
    print(f"Qwen low_samples 总数:   {len(qwen_map)}")
    print(f"共同 sample_id 数:       {len(set(gemini_map.keys()) & set(qwen_map.keys()))}")
    print(f"---")
    print(f"prediction 不一致:       {mismatch_count}")
    print(f"仅在 Gemini 中出现:      {only_gemini_count}")
    print(f"仅在 Qwen 中出现:        {only_qwen_count}")
    print(f"---")
    print(f"结果已导出到: {output_path}")


if __name__ == "__main__":
    main()
