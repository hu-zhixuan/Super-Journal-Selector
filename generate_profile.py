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
{user_draft_text[:2000]}
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

以及该期刊近年发表论文的量化统计聚合指标（包含代表性高引文献与最相似文献）：
```json
{stats_json}
```

{user_draft_section}

请根据以上真实数据，为作者撰写一份严谨、冰冷、无 AI 废话的 Markdown 循证对标与修改诊断报告。

【🔴 极其严厉的输出控制规范】：
1. **必须每句有据**：不准泛泛而谈。提示词中提供的所有结论，凡是涉及到该期刊的方法偏好、样本要求、理论倾向、狙击盲区的判定，**必须在话术中明确用 `《论文标题》` 引用聚合指标里存在的真实文献作为实证！禁止凭空编造证据！**
2. **必须看板化呈现**：拒绝长篇大论的“AI 抒情散文”。必须采用 **“数据表格 + 极简 bullet 点批注”** 的结构。批注要求一针见血，直指痛点。
3. **必须包含以下三大核心部分**：

---

## 一、 目标期刊选稿偏好与近期趋势客观看板 (Objective Preferences & Trend Dashboard)

### 0. 期刊基本学术属性与 SCI / 中科院分区表
用单列表格呈现以下字段：
* 官方 ISSN、H-index 影响力指数、估算影响因子 (Estimated IF)。
* 最新 **JCR 分区**（注明大类与代表性小类分类领域）。
* 最新 **中科院分区**（注明大类分区、代表性小类学科分区、以及是否为 Top 期刊）。

### 1. 研究范式与方法论分布表 (Paradigm & Methodology Distribution)
* 用表格呈现各研究范式的占比与平均被引次数。
* **### 📌 范式解读批注**：字数控制在 100 字内。必须以真实文献（如 `《论文标题》`）举例说明该刊高引范式的特征。

### 2. 理论框架、工具与样本门槛指标表 (Theories, Tools & Sample Thresholds)
* 用表格形式呈现：排名前列的核心理论框架（Top Theories）、常用分析工具链（Top Tools）以及定量分析的样本规模区间（Median/Min/Max）。
* **### 📌 指标解读批注**：用极简短的 2 行字说明样本与工具的隐形门槛。

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

### 2. 🚨 审稿人视角：致命差异狙击点 (Reviewer's Attack Vectors)
请站在最苛刻审稿人视角，用极具攻击性、简短有力的 Bullet 点，指出你的草稿与这 3 篇已录用相似文献之间的 **3 个致命硬伤差在哪里**（例如：样本量相差两个数量级、缺乏物理第一性原理模型推导、完全忽视了某理论构念等）。

---

## 三、 【独家解密】审稿人“出题人”心思与防御性修稿破招 (Preemptive Defense)

### 1. ⏱️ 3分钟初审生死线（AE 心里的淘汰秒表）
用 3 句短话，列出主编在前 180 秒内扫描你的标题摘要、引言框架、图表方程时，会因为什么致命表述立刻做出 Desk Reject 决定。（不准废话）。

### 2. 🛡️ 投稿前预判封口修改方案 (Preemptive Defenses)
针对第二部分发现的致命差异，给出**立竿见影的防御性修改方案**：
* 给出具体的修改前（Before） vs 修改后（After）的标题、摘要或论证句式示例。
* 指明为了堵住审稿人的嘴，你必须在投稿前补上什么图表、稳健性检验或数学物理包装。
"""

        system_prompt = (
            "You are a distinguished, direct-talking Associate Editor. "
            "You write highly structured, evidence-backed diagnostic reports featuring data tables and sharp, citation-driven bullet points in professional Chinese. "
            "You hate robotic filler, verbose AI prose, and empty generalizations. Every claim you make is proved by citing a real paper title."
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
