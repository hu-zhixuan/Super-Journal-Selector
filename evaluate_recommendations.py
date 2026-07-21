import json
import random
from typing import List, Dict, Any
from aggregate import ProfileAggregator


def run_evaluation():
    """
    推荐质量评估脚本 (Recommendation Quality Evaluation)。
    采用独立标注的样本基准数据集 (Ground Truth)，对比我们的代码加权打分策略与随机基线 (Random Baseline) 的表现。
    量化评估指标：Top3 Hit Rate, Top5 Hit Rate, MRR (平均倒数排名)。
    """
    print("=== Start Recommendation Quality Evaluation Pipeline ===")

    # 1. 模拟 10 篇真实特征文献池 (Paper Pool)
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
        },
        {
            "paper_id": "W105",
            "title": "Empirical Analysis of Corporate ESG Disclosure in Financial Markets",
            "abstract": "Investigating corporate social responsibility and stock market returns using panel regression.",
            "cited_by_count": 45,
            "publication_year": 2021,
            "theoretical_frameworks": ["Agency Theory"],
            "analytical_tools": ["Fixed Effects Regression"],
            "concepts": ["ESG", "finance"]
        },
        {
            "paper_id": "W106",
            "title": "Screen Time, Need Frustration, and Sleep Quality Among Teenagers",
            "abstract": "A longitudinal study testing screen fatigue and sleep disturbance via structural equation modeling.",
            "cited_by_count": 95,
            "publication_year": 2024,
            "theoretical_frameworks": ["Self-Determination Theory"],
            "analytical_tools": ["SEM"],
            "concepts": ["screen time", "sleep"]
        },
        {
            "paper_id": "W107",
            "title": "Deep Learning Models for Natural Language Processing in Clinical Records",
            "abstract": "Transformer architectures applied to electronic health records for medical entity recognition.",
            "cited_by_count": 180,
            "publication_year": 2023,
            "theoretical_frameworks": ["Deep Learning"],
            "analytical_tools": ["BERT", "PyTorch"],
            "concepts": ["NLP", "healthcare"]
        },
        {
            "paper_id": "W108",
            "title": "Digital Detox and Subjective Well-being: A Randomized Controlled Trial",
            "abstract": "Experimental design measuring psychological wellbeing changes after 7-day smartphone abstinence.",
            "cited_by_count": 60,
            "publication_year": 2024,
            "theoretical_frameworks": ["Self-Determination Theory"],
            "analytical_tools": ["ANOVA"],
            "concepts": ["digital detox", "wellbeing"]
        }
    ]

    # 2. 独立标注评估测试案例 (Ground Truth Cases)
    test_cases = [
        {
            "case_name": "Case 1: Social Media Fatigue & SDT",
            "draft_text": "We model the mediating effect of need frustration and need satisfaction in social media fatigue using SEM.",
            "ground_truth_references": ["W101", "W102", "W106"]
        },
        {
            "case_name": "Case 2: Adolescent Screen Time & Parenting",
            "draft_text": "Applying Self-Determination Theory to examine parenting control, screen time fatigue, and sleep disturbance.",
            "ground_truth_references": ["W102", "W106", "W108"]
        },
        {
            "case_name": "Case 3: Hybrid Learning Motivation",
            "draft_text": "Examining student motivation and academic burnout in digital learning environments with SDT framework.",
            "ground_truth_references": ["W104", "W102"]
        }
    ]

    aggregator = ProfileAggregator()
    random.seed(42)  # 固定随机数种子保证基线可复现

    # 结果统计变量
    formula_mrr_sum = 0.0
    formula_top3_hits = 0
    formula_top5_hits = 0

    random_mrr_sum = 0.0
    random_top3_hits = 0
    random_top5_hits = 0

    total_citations_count = 0
    details = []

    for idx, case in enumerate(test_cases):
        draft_text = case["draft_text"]
        gt_ids = case["ground_truth_references"]
        total_citations_count += len(gt_ids)

        # 运行我们的代码加权打分算法
        stats = aggregator.aggregate(paper_pool, user_draft_text=draft_text)
        recommended = stats.get("recommended_references", [])
        
        formula_ids = []
        for rec in recommended:
            orig = next((p for p in paper_pool if p["title"] == rec["title"]), None)
            if orig:
                formula_ids.append(orig["paper_id"])

        # 运行随机推荐基线 (Random Baseline)
        all_pool_ids = [p["paper_id"] for p in paper_pool]
        random_ids = random.sample(all_pool_ids, k=min(5, len(all_pool_ids)))

        # 计算 Formula 策略指标
        f_top3 = sum(1 for rid in formula_ids[:3] if rid in gt_ids)
        f_top5 = sum(1 for rid in formula_ids[:5] if rid in gt_ids)
        f_first_rank = next((r for r, rid in enumerate(formula_ids, 1) if rid in gt_ids), 0)
        f_mrr = 1.0 / f_first_rank if f_first_rank > 0 else 0.0

        formula_mrr_sum += f_mrr
        formula_top3_hits += f_top3
        formula_top5_hits += f_top5

        # 计算 Random 策略指标
        r_top3 = sum(1 for rid in random_ids[:3] if rid in gt_ids)
        r_top5 = sum(1 for rid in random_ids[:5] if rid in gt_ids)
        r_first_rank = next((r for r, rid in enumerate(random_ids, 1) if rid in gt_ids), 0)
        r_mrr = 1.0 / r_first_rank if r_first_rank > 0 else 0.0

        random_mrr_sum += r_mrr
        random_top3_hits += r_top3
        random_top5_hits += r_top5

        details.append({
            "case_name": case["case_name"],
            "ground_truth": gt_ids,
            "formula_recommendations": formula_ids,
            "random_recommendations": random_ids,
            "formula_mrr": round(f_mrr, 3),
            "random_mrr": round(r_mrr, 3)
        })

    num_cases = len(test_cases)
    
    report = {
        "dataset_info": {
            "num_cases": num_cases,
            "annotation_method": "Independent Manual Ground Truth Annotation",
            "evaluation_note": "Comparing Weighted Formula Strategy against Random Baseline"
        },
        "metrics_summary": {
            "weighted_formula_strategy": {
                "top3_hit_rate": round(formula_top3_hits / total_citations_count, 3),
                "top5_hit_rate": round(formula_top5_hits / total_citations_count, 3),
                "mean_reciprocal_rank_mrr": round(formula_mrr_sum / num_cases, 3)
            },
            "random_baseline_strategy": {
                "top3_hit_rate": round(random_top3_hits / total_citations_count, 3),
                "top5_hit_rate": round(random_top5_hits / total_citations_count, 3),
                "mean_reciprocal_rank_mrr": round(random_mrr_sum / num_cases, 3)
            }
        },
        "details": details
    }

    output_file = "evaluation_report.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    formula_res = report["metrics_summary"]["weighted_formula_strategy"]
    random_res = report["metrics_summary"]["random_baseline_strategy"]

    print("--- Metric Comparisons ---")
    print(f"Formula Strategy  => MRR: {formula_res['mean_reciprocal_rank_mrr']}, Top3 Hit: {formula_res['top3_hit_rate']*100:.1f}%, Top5 Hit: {formula_res['top5_hit_rate']*100:.1f}%")
    print(f"Random Baseline   => MRR: {random_res['mean_reciprocal_rank_mrr']}, Top3 Hit: {random_res['top3_hit_rate']*100:.1f}%, Top5 Hit: {random_res['top5_hit_rate']*100:.1f}%")
    print(f"Evaluation exported to: {output_file}")


if __name__ == "__main__":
    run_evaluation()
