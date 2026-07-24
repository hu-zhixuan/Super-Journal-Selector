import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from llm_client import LLMClient

logger = logging.getLogger(__name__)


class JournalRouter:
    """
    全网学术期刊智能路由与 Desk Reject 秒拒预测大脑。
    根据用户输入的论文草稿，对比数据库中的多本候选期刊偏好，自动评定"冲刺、主投、保底"三级投递梯队。

    增强版特性：
    - 结合 OpenAlex 实时数据（发文量、影响因子、H-index）作为证据
    - 本地分区数据辅助梯队划分
    - LLM 输出结构校验与自动修复
    - 智能 fallback：LLM 不可用时基于分区数据生成默认推荐
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()
        self._partitions_cache = None

    def _load_partitions(self) -> Dict[str, Any]:
        """加载并缓存本地期刊分区数据库"""
        if self._partitions_cache is not None:
            return self._partitions_cache
        partition_file = os.path.join(os.path.dirname(__file__), "journal_partitions.json")
        if os.path.exists(partition_file):
            try:
                with open(partition_file, "r", encoding="utf-8") as f:
                    self._partitions_cache = json.load(f)
            except Exception as e:
                logger.warning(f"读取期刊数据库失败: {e}")
                self._partitions_cache = {}
        else:
            self._partitions_cache = {}
        return self._partitions_cache

    def _fetch_openalex_metrics(self, journal_name: str) -> Optional[Dict[str, Any]]:
        """从 OpenAlex 获取期刊实时指标（发文量、IF、H-index）"""
        try:
            resp = requests.get(
                "https://api.openalex.org/sources",
                params={"search": journal_name, "per-page": 1},
                proxies={"http": None, "https": None},
                timeout=8,
            )
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                if results:
                    item = results[0]
                    stats = item.get("summary_stats", {})
                    counts = item.get("counts_by_year", [])
                    current_year = datetime.now().year
                    recent_works = sum(
                        c.get("works_count", 0) for c in counts if c.get("year", 0) >= (current_year - 2)
                    )
                    return {
                        "display_name": item.get("display_name", journal_name),
                        "h_index": stats.get("h_index", 0),
                        "estimated_if": round(stats.get("2yr_mean_citedness", 0), 2),
                        "recent_works_2yr": recent_works,
                        "works_count": item.get("works_count", 0),
                    }
        except Exception as e:
            logger.debug(f"OpenAlex 查询 '{journal_name}' 失败: {e}")
        return None

    def _get_tier_by_partition(self, journal_name: str) -> str:
        """根据本地分区数据判断期刊梯队级别"""
        partitions = self._load_partitions()
        q_clean = journal_name.lower().strip()
        for k, v in partitions.items():
            if (k in q_clean) or (q_clean in k):
                cas_zone = v.get("cas_zone", "")
                is_top = v.get("is_top", "")
                if "1区" in cas_zone or "Top" in is_top:
                    return "reaching"
                elif "2区" in cas_zone:
                    return "target"
                else:
                    return "safe"
        return "target"

    def _validate_and_fix_output(self, res_dict: Dict[str, Any], candidate_journals: List[str]) -> Dict[str, Any]:
        """校验 LLM 输出结构，确保包含完整的三级梯队"""
        if not isinstance(res_dict, dict):
            raise ValueError("LLM 输出不是有效字典")

        if "draft_summary_note" not in res_dict:
            res_dict["draft_summary_note"] = "论文主题已解析"
        if "recommended_tiers" not in res_dict or not isinstance(res_dict["recommended_tiers"], list):
            res_dict["recommended_tiers"] = []

        tiers = res_dict["recommended_tiers"]
        valid_tiers = []
        for item in tiers:
            if not isinstance(item, dict):
                continue
            fixed_item = {
                "tier": item.get("tier", "Target (主投)"),
                "journal_name": item.get("journal_name", "Unknown"),
                "fit_score": item.get("fit_score", 75),
                "estimated_acceptance_rate": item.get("estimated_acceptance_rate", "20%-35%"),
                "reason": item.get("reason", "综合评估匹配"),
            }
            try:
                score = int(fixed_item["fit_score"])
                fixed_item["fit_score"] = max(0, min(100, score))
            except (ValueError, TypeError):
                fixed_item["fit_score"] = 75
            valid_tiers.append(fixed_item)

        # 如果 LLM 没有返回完整的三级梯队，基于分区数据自动补齐
        tier_labels = {"reaching": "Reaching (冲刺)", "target": "Target (主投)", "safe": "Safe (保底)"}
        existing_tiers_lower = [t.get("tier", "").lower() for t in valid_tiers]

        if len(valid_tiers) < 3 and candidate_journals:
            partitions = self._load_partitions()
            for tier_key, tier_label in tier_labels.items():
                has_tier = any(tier_key in t for t in existing_tiers_lower)
                if not has_tier:
                    for j in candidate_journals:
                        j_tier = self._get_tier_by_partition(j)
                        if j_tier == tier_key:
                            default_scores = {"reaching": 68, "target": 82, "safe": 91}
                            default_rates = {"reaching": "12%-20%", "target": "25%-35%", "safe": "40%-55%"}
                            zone = partitions.get(j, {}).get("cas_zone", "未知")
                            valid_tiers.append(
                                {
                                    "tier": tier_label,
                                    "journal_name": j.title(),
                                    "fit_score": default_scores[tier_key],
                                    "estimated_acceptance_rate": default_rates[tier_key],
                                    "reason": f"基于中科院分区数据自动补齐 ({zone})",
                                }
                            )
                            break

        res_dict["recommended_tiers"] = valid_tiers
        return res_dict

    def route_journals(self, user_draft_text: str, candidate_journals: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        根据用户草稿与多期刊偏好数据库，输出结构化的多期刊投递阵列。
        融合 OpenAlex 实时指标 + 本地分区证据 + LLM 智能判断。
        """
        if not candidate_journals:
            partitions = self._load_partitions()
            if partitions:
                candidate_journals = list(partitions.keys())
            else:
                candidate_journals = [
                    "computers in human behavior",
                    "computers in human behavior reports",
                    "strategic management journal",
                    "mis quarterly",
                    "physica a",
                ]

        # 获取候选期刊的 OpenAlex 实时指标（作为 LLM 的证据输入）
        journal_evidence = {}
        for j in candidate_journals[:10]:
            metrics = self._fetch_openalex_metrics(j)
            if metrics:
                journal_evidence[j] = metrics

        # 构造带证据的候选列表
        partitions = self._load_partitions()
        evidence_lines = []
        for j in candidate_journals:
            m = journal_evidence.get(j, {})
            zone = partitions.get(j, {}).get("cas_zone", "未知")
            if m:
                evidence_lines.append(
                    f"- {j.title()} | H-index: {m.get('h_index', '?')} | "
                    f"Est.IF: {m.get('estimated_if', '?')} | "
                    f"近2年发文: {m.get('recent_works_2yr', '?')}篇 | "
                    f"分区: {zone}"
                )
            else:
                evidence_lines.append(f"- {j.title()} | 分区: {zone}")

        draft_preview = user_draft_text[:3000] if user_draft_text else ""
        candidate_str = "\n".join(evidence_lines)

        prompt = f"""
You are an expert academic publishing strategist and Journal Selection Director.
Read the user's paper draft preview carefully:

Paper Draft Preview:
```text
{draft_preview}
```

Candidate Journal Pool (with real-time bibliometric evidence):
{candidate_str}

Evaluate the paper draft against the candidate journals using the provided evidence. Recommend the Top 3 best fit journals and classify them into 3 delivery tiers:
1. `Reaching (冲刺)`: High impact/top tier journal (typically CAS Zone 1 / Top), fit score 60%-75%, requires major revisions.
2. `Target (主投)`: Highly aligned paradigm and topic, fit score 80%-90%, optimal primary choice.
3. `Safe (保底)`: High acceptance odds (typically CAS Zone 2-3), fit score > 90%, safe fallback.

IMPORTANT: Use the H-index, Impact Factor, and partition data provided above to make evidence-based tier assignments. Do NOT assign a Zone 3 journal to "Reaching" tier.

Output MUST be clean, valid JSON matching the following structure without markdown formatting:
{{
  "draft_summary_note": "A 1-sentence summary of manuscript core methodology and topic.",
  "recommended_tiers": [
    {{
      "tier": "Reaching (冲刺)",
      "journal_name": "Journal Name",
      "fit_score": 70,
      "estimated_acceptance_rate": "15%-20%",
      "reason": "Concise 1-sentence reason for reaching tier."
    }},
    {{
      "tier": "Target (主投)",
      "journal_name": "Journal Name",
      "fit_score": 85,
      "estimated_acceptance_rate": "25%-35%",
      "reason": "Concise 1-sentence reason for target tier."
    }},
    {{
      "tier": "Safe (保底)",
      "journal_name": "Journal Name",
      "fit_score": 92,
      "estimated_acceptance_rate": "40%-55%",
      "reason": "Concise 1-sentence reason for safe tier."
    }}
  ]
}}
"""
        system_prompt = "You are a professional academic journal selector. Output valid JSON matching the schema only. Base your recommendations on the bibliometric evidence provided."
        try:
            res_dict = self.llm.call_json(prompt=prompt, system_prompt=system_prompt, temperature=0.2)
            validated = self._validate_and_fix_output(res_dict, candidate_journals)
            return validated
        except Exception as e:
            logger.error(f"期刊路由计算失败: {e}")
            # 智能 fallback：基于本地分区数据生成默认三级梯队
            return self._build_fallback_tiers(candidate_journals)

    def _build_fallback_tiers(self, candidate_journals: List[str]) -> Dict[str, Any]:
        """当 LLM 不可用时，基于本地分区数据生成默认三级梯队推荐"""
        fallback_tiers = []
        tier_config = {
            "reaching": ("Reaching (冲刺)", 68, "12%-20%"),
            "target": ("Target (主投)", 82, "25%-35%"),
            "safe": ("Safe (保底)", 91, "40%-55%"),
        }
        partitions = self._load_partitions()
        assigned = set()
        for tier_key, (label, score, rate) in tier_config.items():
            for j in candidate_journals:
                if j not in assigned and self._get_tier_by_partition(j) == tier_key:
                    zone = partitions.get(j, {}).get("cas_zone", "未知")
                    fallback_tiers.append(
                        {
                            "tier": label,
                            "journal_name": j.title(),
                            "fit_score": score,
                            "estimated_acceptance_rate": rate,
                            "reason": f"基于本地分区数据 ({zone}) 的默认推荐",
                        }
                    )
                    assigned.add(j)
                    break
        # 补齐不足的梯队
        if len(fallback_tiers) < 3:
            for j in candidate_journals:
                if j not in assigned:
                    fallback_tiers.append(
                        {
                            "tier": "Target (主投)",
                            "journal_name": j.title(),
                            "fit_score": 78,
                            "estimated_acceptance_rate": "25%-35%",
                            "reason": "默认候选推荐",
                        }
                    )
                    assigned.add(j)
                    if len(fallback_tiers) >= 3:
                        break

        return {
            "draft_summary_note": "（LLM 路由暂时不可用，以下为基于分区数据的默认推荐）",
            "recommended_tiers": fallback_tiers,
        }


if __name__ == "__main__":
    router = JournalRouter()
    res = router.route_journals(
        "We analyze adolescent psychological wellbeing and need frustration under social media fatigue using SEM."
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
