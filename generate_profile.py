import os
import json
import logging
from typing import Dict, Any, Optional
from llm_client import LLMClient

logger = logging.getLogger(__name__)


class ProfileGenerator:
    """
    层④：战略生成层。将层③输出的统计特征、本地对标出来的 Top 3 相似文献、用户待投稿文本，
    拼装为严格“数据表格+极简证据批注”的循证诊断报告。拒绝散文腔调，每句话都必须拿真实论文举证。
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
        基于量化统计、官方指标及相似文献对比，生成高确定性、表格化的对标报告。
        """
        logger.info(f"正在为期刊 '{journal_name}' 撰写循证诊断对比报告...")

        stats_json = json.dumps(aggregated_stats, ensure_ascii=False, indent=2)
        metadata_json = json.dumps(journal_metadata, ensure_ascii=False, indent=2)

        user_draft_section = ""
        if user_draft_text and user_draft_text.strip():
            user_draft_section = f"""
### 【核心重点：拟投稿论文草稿】
用户提交的待诊断论文文本：
```text
{user_draft_text[:100000]}
```

### 【近年发表的 Top 3 最相似文献特征（由 Layer ③ Cosine Similarity 算法计算得出）】
请直接使用这 3 篇真实文献的数据与用户的草稿进行深度维度对比：
{json.dumps(aggregated_stats.get("most_similar_papers", []), ensure_ascii=False, indent=2)}
"""
        else:
            user_draft_section = """
（注：用户本次未提供具体论文草稿。第二部分对比矩阵请以该期刊“典型录用论文”为虚拟靶向进行通用维度基准表对标。）
"""

        prompt = f"""
你是一名在顶尖学术期刊（如 {journal_name}）担任多年编委及高级审稿人（Associate Editor）的资深学者。
下面是数据工程提取的该期刊官方指标：
```json
{metadata_json}
```

以及该期刊近年发表论文在大样本池（通常100篇以上高频聚类）下的量化统计聚合指标（包含代表性高引文献与最相似文献）：
```json
{stats_json}
```

{user_draft_section}

请根据以上大样本真实数据，为作者撰写一份严谨、冰冷、高度具有特殊指向性且无 AI 废话的 Markdown 循证对标与修改诊断报告。
因为我们的指标由大样本聚类算得，你必须充分展现对该刊小同行隐形偏好、真实样本门槛与理论执念的精确洞察！

【🔴 极其严厉的输出控制规范】：
1. **必须每句有据且具有极致特殊指向性**：不准泛泛而谈或使用通用套话。提示词中提供的所有结论，凡是涉及到该期刊的方法偏好、样本底线、理论构念、狙击盲区的判定，**必须在话术中明确用 `《论文标题》` 引用聚合指标里存在的真实文献作为实证！禁止凭空编造证据！**
2. **必须看板化呈现**：拒绝长篇大论的“AI 抒情散文”。必须采用 **“数据表格 + 极简 bullet 点批注”** 的结构。批注要求一针见血，直指痛点。
3. **必须包含以下三大核心部分**：

---

## 一、 目标期刊选稿偏好与近期趋势客观看板 (Objective Preferences & Trend Dashboard)

### 0. 期刊基本学术属性与 JCR / 中科院分区表 (由本地权威数据映射)
用单列表格或键值对表格呈现以下字段，数据请严格提取自 `metadata_json` 中的对应键值（切勿幻觉或编造）：
* 官方 ISSN、H-index 影响力指数、估算影响因子 (Estimated IF)。
* 最新 **JCR 分区** (如 Q1, Q2，对应 `jcr_zone`)。
* 最新 **中科院分区** (如 1区, 2区, 3区，对应 `cas_zone`，小类划分对应 `cas_sub_categories`，是否为 Top 期刊对应 `is_top`)。

### 1. 研究范式与方法论分布表 (Paradigm & Methodology Distribution)
* 用表格呈现各研究范式的占比与平均被引次数。
* **### 📌 范式解读批注**：字数控制在 100 字内。必须以真实文献（如 `《论文标题》`）举例说明该刊高引范式的特征。

### 2. 理论框架、工具与样本门槛指标表 (Theories, Tools & Sample Thresholds)
* 用表格形式呈现：排名前列的核心理论框架（Top Theories）、常用分析工具链（Top Tools）以及定量分析的样本规模区间（Median/Min/Max）。
* **### 📌 指标解读批注**：用极简短的 2 行字说明样本与工具的隐形门槛。
  特别注意：如果该刊为物理学或自然科学期刊且统计出的样本量中位数（Median Sample）极小（如 1 或 2），说明此非社会科学人头调查，而是实验器件样品、单晶体系或模型层数。请务必在批注中以高级审稿人身份明确对此范式差异做出科学合理解释，避免误导作者。

### 3. 开源科学实践与统计汇报规范底线 (Open Science & Statistical Norms Audit)
* 根据 `stats_json` 中的 `open_science_stats` 用表格呈现数据开源（Open Data）与代码开源（Open Code）等实践在近年文章中的实际占比，判断该刊是否属于开源友好型期刊。
* 根据 `stats_json` 中的 `top_reporting_styles` 展现该刊的假设检验与统计报告风格偏好排行（例如是否普遍要求 Bootstrap 中介效应检验、显著性区间汇报等）。
* **### 📌 开源与统计审计批注**：以 AE 身份用 1 句话点破该刊对数据公开或方法透明度的底线态度。

---

## 二、 【核心杀手锏】用户稿件 vs 近年最相似的 3 篇已发表文献深度对比矩阵 (Contrastive Diagnosis)
*(若用户提供了草稿，请将草稿与 Layer ③ 算出的 3 篇相似文献做严格的矩阵对标；若未提供草稿，则与典型已发表论文做对比)*

### 1. 维度对比对标矩阵表
请构建一个 Markdown 表格，对比以下列：
`对标维度` | `你的论文草稿` | `相似文献1:《标题》 (IF/被引)` | `相似文献2:《标题》 (IF/被引)` | `相似文献3:《标题》 (IF/被引)`
表格行必须包含：
* **核心研究范式** (定量实证/定性案例/混合方法/理论推导/计算仿真)
* **核心理论视角** (引入的理论框架与核心构念)
* **样本规模与数据来源** (具体数据源与 N 的量级)
* **核心分析工具与模型** (使用的计量模型、统计方法或机器学习工具)
* **创新贡献定位** (如何向主编写故事/贡献点落脚处)

### 2. 🔍 审稿人视角：核心差距诊断 (Reviewer's Objective Assessment)
请站在客观、严谨且富有建设性的审稿人视角，用事实说话，指出你的草稿与这 3 篇已录用相似文献之间的 **3 个核心差距与不足**（如：模型复杂度的理论支撑差异、样本边界条件限制、或是分析方法的严密程度等）。请客观判断，区分“致命硬伤”（如方法错误）与“可改进项”（如补做控制或解释），切忌为了批评而刻意全盘否定。

---

## 三、 【独家解密】审稿人关注焦点与防御性修稿策略 (Preemptive Defense)

### 1. ⏱️ 3分钟初审过滤器（AE 心中的关键评估项）
用 3 句短话，列出主编在前 180 秒内扫描你的标题摘要、引言框架和核心方法时，最容易触发 Desk Reject 的几个**客观硬伤**（如研究问题过时、理论贡献陈述不清等），提供客观警示。

### 2. 🛡️ 投稿前防御性修改方案 (Actionable Defenses)
针对第二部分发现的核心差距，给出**切实可行的防御性修改建议**：
* 给出具体的修改前（Before） vs 修改后（After）的标题、摘要或论证句式示例。
* 指明如何通过补充实验、细化方法描述、或在讨论部分合理增加限制说明（Limitation）来堵住审稿人的质疑，帮助稿件达到录用门槛。
"""

        system_prompt = (
            "You are a fair, objective, and highly constructive Associate Editor. "
            "You write highly structured, evidence-backed diagnostic reports featuring data tables and professional, balanced academic advice in Chinese. "
            "You aim to point out real gaps based on empirical statistics while offering actionable paths for revision. Every claim you make is proved by citing a real paper title."
        )

        try:
            report_content = self.llm.call(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.18, # 极低温度，防止大模型自由编造散文
                max_retries=3,
            )
            logger.info("对标诊断报告生成完毕。")
            return report_content
        except Exception as e:
            logger.error(f"生成报告异常: {str(e)}")
            raise
