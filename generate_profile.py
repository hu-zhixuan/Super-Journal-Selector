import os
import json
import logging
from typing import Dict, Any, Optional
from llm_client import LLMClient

logger = logging.getLogger(__name__)


class ProfileGenerator:
    """
    层④：战略生成层。将层③输出的量化统计特征，与 OpenAlex 期刊基本指标、用户待投稿文本进行提示词工程拼装，
    利用 LLM 的高级学术推理与写作经验，生成客观严谨、包含 SCI/中科院分区的深度《期刊偏好画像与修稿策略报告》。
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    def generate_report(
        self,
        journal_name: str,
        aggregated_stats: Dict[str, Any],
        journal_metadata: Dict[str, Any],
        user_draft_text: Optional[str] = None,
    ) -> str:
        """
        基于统计聚合指标、期刊官方指标与可选的用户草稿文本，生成 Markdown 画像与修稿策略
        """
        logger.info(f"正在为期刊 '{journal_name}' 撰写深度画像与策略报告...")

        stats_json = json.dumps(aggregated_stats, ensure_ascii=False, indent=2)
        metadata_json = json.dumps(journal_metadata, ensure_ascii=False, indent=2)

        user_draft_section = ""
        if user_draft_text and user_draft_text.strip():
            user_draft_section = f"""
### 【核心重点：用户拟投稿论文摘要/大纲对标诊断】
用户的论文文本（摘要/大纲/草稿部分）如下：
```text
{user_draft_text[:3000]}
```
请务必结合前面的期刊选稿偏好统计指标与分区特性，在第三部分对用户的这段文本做出专业、细致、具体的对标优化建议与改写指导。
"""
        else:
            user_draft_section = """
（注：当前用户未提供具体投稿摘要草稿。在第三部分请从通用对标修稿策略出发，给出该刊典型的标题重构模式、Abstract 规范化结构建议及关键论证补强模板。）
"""

        prompt = f"""
你是一名在顶尖学术期刊（如 {journal_name}）担任多年编委及高级审稿人（Associate Editor）的资深学者。
现在我们通过数据工程提取了该期刊的官方元数据指标：
```json
{metadata_json}
```

以及该期刊近年发表论文的量化统计聚合指标：
```json
{stats_json}
```

{user_draft_section}

请根据以上数据，为准备向 `{journal_name}` 投稿的科研人员输出一份专业、客观、深度、逻辑严密且指导性极强的 Markdown 画像与策略建议报告。
报告必须严格包含以下三大核心部分（需用严谨规范的学术话语阐述，兼顾深度与实用性）：

## 一、 目标期刊选稿偏好核心画像 (Journal Preference Profile)

### 0. 期刊基本学术属性与 SCI / 中科院分区评估
请结合给出的期刊元数据（ISSN, H-index, 估算影响因子等），依托你的前沿学术数据库知识，给出以下核心指标的权威对标（必须明确写出）：
- **期刊收录属性**：标明是否为 SCI, SCIE, SSCI, AHCI 或 ESCI 收录。
- **JCR 分区**：指出该期刊在 JCR 中的官方分区等级 (Q1, Q2, Q3 或 Q4) 及其主要分类领域。
- **中科院分区 (CAS Partition)**：指出该期刊在最新版“中科院期刊分区表”中的**大类学科分区**（一区/二区/三区/四区，Top期刊属性）及**代表性小类学科分区**。
- **影响力简评**：结合 H-index 和 影响因子 (Impact Factor) 评估该刊在所在细分领域内的学术地位与投递难度。

### 1. 范式与方法论倾向量化解读
深入分析各种研究方法占比数据。说明该刊在定量实证、定性案例、混合研究或计算社会科学/AI仿真等新范式上的偏好分布，以及不同研究范式的均次被引情况。

### 2. 理论框架与研究视角矩阵
归纳总结该刊高频出现的理论视角（Top Theories），并建议作者在文献综述与理论构建时着重关注的核心理论话语场域。

### 3. 样本量级与分析工具要求
针对实证或数据分析类研究，梳理该刊在有效样本量（最小值、中位数以及期望门槛）以及统计计量/机器学习分析工具链上的特征要求。

### 4. 近期热点文章与创新点剖析
结合高被引代表性研究，总结该刊近年来在选题新颖性与研究贡献上的典型风向。

## 二、 常见避坑指南与审稿雷区 (Desk Reject Red Flags)
基于统计结果中极低占比的方法类型或潜在的数据学短板，提出 3~4 个需要特别规避的审稿雷区（例如样本量不足、分析方法单一、理论构建不足等），并说明何种类型的研究容易在初审（Desk Reject）阶段遇阻。

## 三、 定制化投稿对标与修稿策略 (Actionable Tailoring Strategy)
{"请直接针对用户提供的上述论文草稿，给出建设性、针对性的诊断意见与改进方案（包括建议优化后的新标题、摘要重构框架与实证补充指导）：" if user_draft_text else "请给出向该期刊投稿时的实用修稿策略与优化建议规范："}
1. **标题与摘要（Title & Abstract）重构技巧**：说明如何优化标题与摘要的结构，使其精准契合该刊的学术风格与重点关切，并结合实例或模板展示。
2. **引言与动机（Introduction & Motivation）强化**：指导如何构建清晰的研究鸿沟（Research Gap），突出研究的必要性与重要性。
3. **方法论与稳健性（Methodological Robustness）建议**：针对实证检验细节或数据处理分析，提出有助于提升论文严谨度与说服力的具体补充方案。

排版请使用规范清晰的 Markdown 格式，合理利用表格、列表与加粗标注，使整份报告结构清晰、内容扎实、便于学者阅读与实操使用。
"""

        system_prompt = (
            "You are a distinguished Associate Editor and quantitative research analyst at a leading academic journal. "
            "You provide insightful, rigorous, objective, and highly professional journal profiles and revision strategies in clear academic Chinese."
        )

        try:
            report_content = self.llm.call(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.35,
                max_retries=3,
            )
            logger.info("画像与策略建议报告生成完毕。")
            return report_content
        except Exception as e:
            logger.error(f"生成报告异常: {str(e)}")
            raise
