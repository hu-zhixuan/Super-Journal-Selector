import json
import re
import os
from typing import List, Dict, Any
from aggregate import ProfileAggregator


def run_evaluation():
    """
    推荐质量评估脚本 (Recommendation Quality Evaluation)。
    内置小规模人工标注 Ground Truth 测试集，评估推荐加权排序算法在对标对齐场景下的效果。
    量化指标：Top3 Hit Rate, Top5 Hit Rate, MRR (平均倒数排名)。
    """
    print("=== Start Recommendation Quality Evaluation Pipeline ===")

    # 1. 模拟特征文献池 (Paper Pool)
    paper_pool = [
        {
            "paper_id": "W101",
            "title": "Adolescent Social Media Fatigue and Depression: A Self-Determination Perspective",
            "abstract": "We analyze adolescent psychological wellbeing and need frustration path under social media fatigue using SmartPLS SEM.",
            "cited_by_count": 120,
            "publication_year": 2024,
            "theoretical_frameworks": ["Self-Determination Theory", "SDT"],
            "analytical_tools": ["SEM", "SmartPLS"],
            "concepts": ["social media fatigue", "wellbeing"]
        },
        {
            "paper_id": "W102",
            "title": "Parental Control and Adolescent Need Satisfaction in Digital Media Era",
            "abstract": "We survey N=300 high school students and run mediation analysis in SmartPLS for need satisfaction and parenting styles.",
            "cited_by_count": 80,
            "publication_year": 2023,
            "theoretical_frameworks": ["Self-Determination Theory"],
            "analytical_tools": ["Regression", "SmartPLS"],
            "concepts": ["parenting", "need satisfaction"]
        },
        {
            "paper_id": "W103",
            "title": "Quantum Physics and Semiconductor Fabrication in Nanotechnology",
            "abstract": "This study analyzes gallium nitride semiconductor physical properties and photonics simulation models.",
            "cited_by_count": 250,
            "publication_year": 2022,
            "theoretical_frameworks": ["Quantum Theory"],
            "analytical_tools": ["Simulation"],
            "concepts": ["nanotechnology", "semiconductor"]
        },
        {
            "paper_id": "W104",
            "title": "Self-Determination and Student Motivation in Hybrid Learning Environments",
            "abstract": "Applying SDT to investigate how hybrid classroom needs satisfaction mediates academic self-efficacy and learning burnout.",
            "cited_by_count": 10,
            "publication_year": 2024,
            "theoretical_frameworks": ["Self-Determination Theory"],
            "analytical_tools": ["Structural Equation Modeling"],
            "concepts": ["motivation", "academic well-being"]
        }
    ]

    # 2. 人工标注评估集 (Draft Ground Truth Cases)
    # 模拟用户拟投草稿及标注应召回文献
    test_cases = [
        {
            "draft_text": "We model the mediating effect of need frustration and need satisfaction in social media fatigue using SEM.",
            "ground_truth_references": ["W101", "W102"]  # 对应 SDT 心理/社交媒体疲劳文章
        },
        {
            "draft_text": "Applying Self-Determination Theory to examine high school motivation and need satisfaction path.",
            "ground_truth_references": ["W102", "W104"]  # 对应教育/动机 SDT 文章
        }
    ]

    aggregator = ProfileAggregator()
    
    total_mrr = 0.0
    top3_hits = 0
    top5_hits = 0
    total_eval_citations = 0

    results_details = []

    for idx, case in enumerate(test_cases):
        draft_text = case["draft_text"]
        gt_ids = case["ground_truth_references"]
        total_eval_citations += len(gt_ids)

        # 运行 Layer ③ 加权排序公式算出得分
        stats = aggregator.aggregate(paper_pool, user_draft_text=draft_text)
        recommended = stats.get("recommended_references", [])
        
        # 对应提取推荐列表标题找到原始 paper_id
        recommended_ids = []
        for rec in recommended:
            orig = next((p for p in paper_pool if p["title"] == rec["title"]), None)
            if orig:
                recommended_ids.append(orig["paper_id"])

        # 计算 Hit Rate
        hits_in_top3 = 0
        hits_in_top5 = 0
        
        first_hit_rank = 0
        for rank, rec_id in enumerate(recommended_ids, 1):
            if rec_id in gt_ids:
                if rank <= 3:
                    hits_in_top3 += 1
                if rank <= 5:
                    hits_in_top5 += 1
                if first_hit_rank == 0:
                    first_hit_rank = rank

        top3_hits += hits_in_top3
        top5_hits += hits_in_top5
        
        # 计算倒数排名 (MRR)
        mrr = 1.0 / first_hit_rank if first_hit_rank > 0 else 0.0
        total_mrr += mrr

        results_details.append({
            "case_idx": idx + 1,
            "draft_preview": draft_text[:50] + "...",
            "ground_truth": gt_ids,
            "recommended": recommended_ids,
            "mrr": round(mrr, 3),
            "hits_top3": hits_in_top3,
            "hits_top5": hits_in_top5
        })

    num_cases = len(test_cases)
    mean_mrr = total_mrr / num_cases if num_cases > 0 else 0.0
    top3_hit_rate = top3_hits / total_eval_citations if total_eval_citations > 0 else 0.0
    top5_hit_rate = top5_hits / total_eval_citations if total_eval_citations > 0 else 0.0

    report = {
        "evaluation_summary": {
            "total_test_cases": num_cases,
            "total_annotated_citations": total_eval_citations,
            "top3_hit_rate": round(top3_hit_rate, 3),
            "top5_hit_rate": round(top5_hit_rate, 3),
            "mean_reciprocal_rank_mrr": round(mean_mrr, 3)
        },
        "details": results_details
    }

    # 结果落盘保存
    output_file = "evaluation_report.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Evaluation Finished. Test set MRR Mean: {report['evaluation_summary']['mean_reciprocal_rank_mrr']}")
    print(f"Top 3 Hit Rate: {report['evaluation_summary']['top3_hit_rate'] * 100:.1f}%")
    print(f"Top 5 Hit Rate: {report['evaluation_summary']['top5_hit_rate'] * 100:.1f}%")
    print(f"Evaluation report exported to: {output_file}")


if __name__ == "__main__":
    run_evaluation()
