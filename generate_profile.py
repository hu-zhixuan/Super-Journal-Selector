import json
import logging
from typing import Dict, Any, Optional
from llm_client import LLMClient

logger = logging.getLogger(__name__)


class ProfileGenerator:
    """
    层④：战略生成层。将层③输出的量化统计特征，与用户待投稿文本/目标进行提示词工程拼装，
    利用 LLM 的高级推理与学术写作经验，生成深度的期刊品味画像与定制化改造策略报告。
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    def generate_report(
        self,
        journal_name: str,
        aggregated_stats: Dict[str, Any],
        user_draft_text: Optional[str] = None,
    ) -> str:
        """
        基于统计聚合指标与可选的用户草稿文本，生成 Markdown 画像与修稿策略
        """
        logger.info(f"正在为期刊 '{journal_name}' 撰写深度画像与策略报告...")

        stats_json = json.dumps(aggregated_stats, ensure_ascii=False, indent=2)

        user_draft_section = ""
        if user_draft_text and user_draft_text.strip():
            user_draft_section = f"""
### 【核心重点：用户拟投稿论文摘要/大纲对标对齐】
用户的论文文本（摘要/大纲/草稿部分）如下：
```text
{user_draft_text[:3000]}
```
请务必结合前面的期刊选稿偏好统计指标，在第三部分对用户的这段文本做出字面级别的针对性改造指导！
"""
        else:
            user_draft_section = """
（注：当前用户未提供具体投稿摘要草稿。在第三部分请从通用对标修稿策略出发，给出该刊典型的标题重构模式、Abstract 黄金三段论结构建议及关键论证补强模板。）
"""

        prompt = f"""
你是一名在顶尖学术期刊（如 {journal_name}）担任多年编委及高级审稿人（Associate Editor）的资深科学家。
现在我们通过纯工程和 LLM 提取了该期刊最近发表论文的量化统计聚合指标，数据如下：

```json
{stats_json}
```

{user_draft_section}

请根据以上真实的量化聚合数据，为准备向 `{journal_name}` 投稿的学者输出一份权威、深度、洞察力极强、排版精美的高质量 Markdown 画像与策略指导报告。
报告必须严格包含以下三大核心部分（不可缺失，需用专业学术话语阐述）：

## 一、 目标期刊选稿偏好核心画像 (Journal Preference Profile)
1. **范式与方法论倾向量化解读**：深入分析各种方法学占比数据。说明该刊是偏向实证模型、案例理论构建，还是人工智能与计算社会科学等新范式，以及哪类范式具有最高的被引期望。
2. **理论框架与研究视角矩阵**：点评该刊高频出现的理论视角（Top Theories），并指出作者应在 Literature Review 中聚焦的主流理论对话场域。
3. **样本阈值与分析工具门槛**：明确给出该刊实证研究在样本量级（最小/中位数/期望）以及统计计量工具（如 SEM, DID, Machine Learning）上的底线硬要求。
4. **近期爆款文章核心创新特征剖析**：结合高被引代表性金句，提炼该刊最近青睐的“破圈”研究品味。

## 二、 常见避坑指南与审稿雷区 (Desk Reject Red Flags)
基于聚合结果中占比极低或极易发生致命缺陷的方法/视角，给出 3~4 个致命雷区（例如样本量过小、方法工具老化、缺乏理论创新包装等），指明什么样样式的稿件会被主编（Editor-in-Chief）直接初审拒稿（Desk Reject）。

## 三、 定制化投稿对标与修稿策略 (Actionable Tailoring Strategy)
{"请直接针对用户提供的上述论文摘要草稿，进行诊断并给出修改方案（包括建议优化后的新标题、新摘要框架与实证补充意见）！" if user_draft_text else "给出向该期刊投稿时的通用修稿实战模板与高通关技巧："}
1. **标题与摘要（Title & Abstract）包装重构技巧**：展示如何把常见的平铺直叙式摘要改造为符合该刊语感的高穿透力结构。
2. **引言与动机（Introduction & Motivation）对齐点**：如何建立能够触动该刊主编/审稿人的研究鸿沟（Research Gap）。
3. **方法论与robustness建议**：需增加哪些稳健性检验或工具链升级，以顺利跨过该刊的方法论审稿门槛。

排版请使用优雅的 Markdown 格式，适当运用重点加粗和表格/引用块让报告一目了然、极具指导价值！
"""

        system_prompt = (
            "You are a top academic editor and quantitative research analyst. "
            "You write insightful, rigorous, data-backed journal profiles and revision strategies in professional Chinese."
        )

        try:
            report_content = self.llm.call(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.4,
                max_retries=3,
            )
            logger.info("画像报告生成完毕。")
            return report_content
        except Exception as e:
            logger.error(f"生成最终画像报告异常: {str(e)}")
            raise
