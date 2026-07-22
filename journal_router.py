import os
import json
import logging
from typing import Dict, Any, List, Optional
from llm_client import LLMClient

logger = logging.getLogger(__name__)


class JournalRouter:
    """
    全网学术期刊智能路由与 Desk Reject 秒拒预测大脑。
    根据用户输入的论文草稿，对比数据库中的多本候选期刊偏好，自动评定“冲刺、主投、保底”三级投递梯队。
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    def route_journals(
        self,
        user_draft_text: str,
        candidate_journals: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        根据用户草稿与多期刊偏好数据库，输出结构化的多期刊投递阵列。
        """
        if not candidate_journals:
            # 从本地 journal_partitions.json 中读取内置候选期刊库
            partition_file = os.path.join(os.path.dirname(__file__), "journal_partitions.json")
            if os.path.exists(partition_file):
                try:
                    with open(partition_file, "r", encoding="utf-8") as f:
                        partitions = json.load(f)
                    candidate_journals = list(partitions.keys())
                except Exception as e:
                    logger.warning(f"读取期刊数据库失败: {e}")
                    candidate_journals = [
                        "computers in human behavior",
                        "computers in human behavior reports",
                        "strategic management journal",
                        "mis quarterly",
                        "physica a"
                    ]
            else:
                candidate_journals = [
                    "computers in human behavior",
                    "computers in human behavior reports",
                    "strategic management journal"
                ]

        draft_preview = user_draft_text[:3000] if user_draft_text else ""
        candidate_str = ", ".join([j.title() for j in candidate_journals])

        prompt = f"""
You are an expert academic publishing strategist and Journal Selection Director.
Read the user's paper draft preview carefully:

Paper Draft Preview:
```text
{draft_preview}
```

Candidate Journal Pool:
{candidate_str}

Evaluate the paper draft against the candidate journals. Recommend the Top 3 best fit journals and classify them into 3 delivery tiers:
1. `Reaching` (冲刺期刊): High impact/top tier journal, fit score 60%-75%, requires major revisions to meet standards.
2. `Target` (主投期刊): Highly aligned paradigm and topic, fit score 80%-90%, optimal primary choice.
3. `Safe` (保底期刊): High acceptance odds, fit score > 90%, safe fallback choice.

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
        system_prompt = "You are a professional academic journal selector. Output valid JSON matching the schema only."
        try:
            res_dict = self.llm.call_json(prompt=prompt, system_prompt=system_prompt, temperature=0.2)
            return res_dict
        except Exception as e:
            logger.error(f"期刊路由计算失败: {e}")
            return {
                "draft_summary_note": "未识别出草稿主题",
                "recommended_tiers": [
                    {
                        "tier": "Target (主投)",
                        "journal_name": candidate_journals[0].title() if candidate_journals else "Target Journal",
                        "fit_score": 80,
                        "estimated_acceptance_rate": "25%-35%",
                        "reason": "基准评估默认匹配"
                    }
                ]
            }


if __name__ == "__main__":
    router = JournalRouter()
    res = router.route_journals("We analyze adolescent psychological wellbeing and need frustration under social media fatigue using SEM.")
    print(json.dumps(res, ensure_ascii=False, indent=2))
