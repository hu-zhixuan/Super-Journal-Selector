import os
import json
import logging
import re
from typing import Dict, Any, Optional
from llm_client import LLMClient

logger = logging.getLogger(__name__)


class ProfileGenerator:
    """
    层④：战略生成层。将层③输出的统计特征、本地对标出来的 Top 3 相似文献、Top 5 推荐引用文献以及用户草稿，
    拼装为“数据表格+极简证据批注”的循证诊断报告。
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
        基于量化统计、官方指标及相似文献对比，生成高确定性、表格化的对标报告，并自动运行 Citation Validator 进行引用强校验。
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
{user_draft_text[:150000]}
```

### 【近年发表的 Top 3 最相似文献特征（由 Layer ③ Cosine Similarity 算法计算得出）】
请直接使用这 3 篇真实文献的数据与用户的草稿进行深度维度对比：
{json.dumps(aggregated_stats.get("most_similar_papers", []), ensure_ascii=False, indent=2)}

### 【近年高契合度推荐引用文献列表（由 Layer ③ 代码打分排序生成，禁止编造）】
以下文献是由代码公式计算出的最适合引用并用来增强你论文的文献，请直接对这 5 篇文献进行点评并说明建议引用理由：
{json.dumps(aggregated_stats.get("recommended_references", []), ensure_ascii=False, indent=2)}
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

【🔴 极其严厉的输出与去 AI 味规范】：
0. **绝对禁止任何问候语与寒暄开场白**：严禁输出“好的，作者”、“作为...编委”、“我已基于大样本数据完成了诊断”等任何口水话与 AI 抒情句！直接从 `# 《期刊名》选稿偏好与循证对标诊断报告` 开始输出正文！
1. **必须每句有据且具有极致特殊指向性**：不准泛泛而谈或使用通用套话。提示词中提供的所有结论，凡是涉及到该期刊的方法偏好、样本底线、理论构念、修稿盲区的判定，**必须在话术中明确用 `《论文标题》` 引用提供给你的真实文献作为实证！禁止凭空编造或幻想任何文献！**
2. **必须看板化呈现**：拒绝长篇大论的“AI 抒情散文”。必须采用 **“数据表格 + 极简 bullet 点批注”** 的结构。批注要求一针见血，直指痛点。
3. **对标文献与推荐引用文献必须分开**：
   * **Top 3 最相似文献**：用来进行稿件设计上的维度差距诊断；
   * **Top 5 推荐引用文献**：用来增强文献综述和讨论部分，并提供具体的“建议引用理由”。你只能对输入给你的这 5 篇文献写建议，禁止捏造其它文献！
4. **必须包含以下三大核心部分**：

---

## 一、 目标期刊选稿偏好与近期趋势客观看板 (Objective Preferences & Trend Dashboard)

### 0. 期刊基本学术属性与 JCR / 中科院分区表 (由本地权威数据映射)
用单列表格呈现以下字段，数据请严格提取自 `metadata_json` 中的对应键值（切勿幻觉或编造）：
* 官方 ISSN、H-index 影响力指数、估算影响因子 (Estimated IF)。
* 最新 **JCR 分区** (如 Q1, Q2，对应 `jcr_zone`)。
* 最新 **中科院分区** (如 1区, 2区, 3区，对应 `cas_zone`，小类划分对应 `cas_sub_categories`，是否为 Top 期刊对应 `is_top`)。

### 1. 研究范式与方法论分布表 (Paradigm & Methodology Distribution)
* 用表格呈现各研究范式的占比与平均被引次数。
* **### 📌 范式解读批注**：字数控制在 100 字内。必须以真实文献（如 `《论文标题》`）举例说明该刊高引范式的特征。

### 2. 理论框架、工具与样本门槛指标表 (Theories, Tools & Sample Thresholds)
* 用表格形式呈现：排名前列的核心理论框架（Top Theories）、常用分析工具链（Top Tools）以及定量分析的样本规模区间（Median/Min/Max）。
* **### 📌 指标解读批注**：用极简短的 2 行字说明样本与工具的隐形门槛。

### 3. 开源科学实践与统计汇报规范底线 (Open Science & Statistical Norms Audit)
* 根据 `stats_json` 中的 `open_science_stats` 用表格呈现数据开源（Open Data）与代码开源（Open Code）等实践在近年文章中的实际占比，判断该刊是否属于开源友好型期刊。
* 根据 `stats_json` 中的 `top_reporting_styles` 展现该刊的假设检验与统计报告风格偏好排行（例如是否普遍要求 Bootstrap 中介效应检验、显著性区间汇报等）。
* **### 📌 开源与统计审计批注**：以 AE 身份用 1 句话点破该刊对数据公开或方法透明度的底线态度。

### 🚨 4. 【核心死穴预测】主编 Desk Reject (秒拒) 致命红线审计卡 (Desk Reject Predictor)
请对比该期刊近 3 年大样本范式与样本门槛，列出 2 条可能导致主编在提交后 48 小时内直接秒拒 (Desk Reject) 的物理硬伤（例如：范式严重偏离期刊非主流、样本量低于绝大多数录用文章、缺少该刊偏好的核心理论）：
* **🚨 Desk Reject 红线 1**：具体判定理由与证据说明。
* **🚨 Desk Reject 红线 2**：具体判定理由与证据说明。

---

## 二、 【核心杀手锏】用户稿件 vs 近年最相似文献对标及推荐引用 (Contrastive Diagnosis)

### 1. 维度对比对标矩阵表
请构建一个 Markdown 表格，对比以下列：
`对标维度` | `你的论文草稿` | `相似文献1:《标题》` | `相似文献2:《标题》` | `相似文献3:《标题》`
（注：相似文献请填入 `most_similar_papers` 中的 Top 3 论文）

### 🔬 2. Top 1 标杆文献与你的草稿：段落级逻辑链解构图谱 (Paragraph-Level Logical Alignment)
请提取 `most_similar_papers` 中排名第 1 的顶级标杆论文，构建一个【段落级逻辑链拆解与草稿差距对齐表】：
| 逻辑段落步骤 | Top 1 标杆论文《标题》的写作逻辑骨架 | 你的论文草稿对应表现 | 像素级修改建议 |
| :--- | :--- | :--- | :--- |
| **步骤 1: 现象引入与痛点** | 标杆如何从行业现实现象引出学术争议 | 你的草稿引言逻辑 | 具体句式重构建议 |
| **步骤 2: 理论冲突与黑箱** | 标杆如何利用理论构建逻辑冲突与矛盾 | 你的草稿理论逻辑 | 具体理论视角补充建议 |
| **步骤 3: 方法与数据支撑** | 标杆如何展示数据代表性与方法论严谨性 | 你的草稿数据描述 | 具体的实验/样本补齐建议 |
| **步骤 4: 学术增量与贡献** | 标杆如何总结 theoretical & practical contributions | 你的草稿贡献总结 | 具体贡献提升声明建议 |

### 3. 🎯 推荐引用的本刊近年高契合度文献列表 (Top 5 Recommended References to Cite)
请根据 `recommended_references` 中的 5 篇论文，构建一个表格：
`序号` | `推荐论文标题` | `发表年份` | `总被引用数` | `最终匹配得分` | `建议引用理由`
（注意：“建议引用理由”由你撰写，说明该论文如何能在论证逻辑、测量工具、或对照样本上帮助增强用户的论文）

### 3. 🌟 稿件核心亮点与学术增量 (Objective Strengths & Contributions)
请客观、不带偏见地指出用户的论文草稿相比于这 3 篇相似文献，有哪些独特的长处、新颖视角或特定的应用价值，保证评议的公正性。

### 4. 🔍 审稿人视角：核心差距与局限性诊断 (Balanced Assessment of Gaps)
客观指出其合理性，切勿无脑唱衰。只有在确实存在方法论漏洞、样本量严重不足支撑模型、或理论增量极度匮乏时，才判定为“关键局限性”并给出修改方向。

---

## 三、 【独家解密】审稿人关注焦点与防御性修稿策略 (Preemptive Defense)

### 1. ⏱️ 3分钟初审过滤器（AE 心中的关键评估项）
用 3 句短话，列出主编在前 180 秒内扫描你的标题摘要、引言框架和核心方法时，最容易触发 Desk Reject 的几个客观硬伤。

### 2. 🛡️ 投稿前防御性修改方案 (Actionable Defenses)
针对第二部分发现的核心差距，给出切实可行的防御性修改建议，并给出修改前（Before） vs 修改后（After）的标题、摘要或论证句式示例。

---

## 四、 【客观预测】发表录用概率评估与条件提升路径 (Acceptance Probability & Elevation Roadmap)

### 1. 🎯 稿件录用概率量化评估看板 (Acceptance Probability Assessment)
请构建一个表格，给出客观公正的量化评估：
| 评估维度 | 指标参数 / 评估结论 |
| :--- | :--- |
| **目标期刊近年基准录用率** | 提取该期刊学术梯队的客观录用难度估计 (通常 15% ~ 25%) |
| **稿件范式与数据契合度得分** | 综合范式匹配、样本量达标情况计算对标得分 (0 - 100 分) |
| **当前草稿状态预估录用概率** | 综合估计当前未经修改直接投稿的录用概率区间 (如 20% - 30%) |
| **完成针对性修改后预估概率** | 严格按照第三/四部分建议修改防御后的预测录用概率区间 (如 55% - 70%) |

### 2. ⚠️ 影响录用概率的核心扣分硬伤 (Critical Risk Factors)
* 用 2 - 3 句短话，指出当前稿件在样本规模、研究范式或理论深度上最拉低录用概率的硬性缺陷。

### 3. 📈 录用概率阶梯提升条件清单 (Step-by-Step Probability Boost Checklist)
列出 3 条明确的“条件式提升路径”，说明满足哪些修改条件可以实现录用概率的跨越：
* **条件 A（预测概率提升 +15% ~ +20%）**：如补齐样本量至期刊中位数线（N ≥ 对应中位数），或补充特定的对比实验；
* **条件 B（预测概率提升 +10% ~ +15%）**：如按建议在论文中引用并对话 Top 5 推荐列表中的本刊近年论文；
* **条件 C（预测概率提升 +5% ~ +10%）**：如规范统计汇报格式（如汇报 Bootstrap 中介效应置信区间与效应量）。
"""

        system_prompt = (
            "You are a fair, objective, and highly constructive Associate Editor. "
            "You write highly structured, evidence-backed diagnostic reports featuring data tables and professional, balanced academic advice in Chinese. "
            "You aim to point out real gaps based on empirical statistics while offering actionable paths for revision. "
            "Do not over-criticize; recognize the manuscript's unique strengths and ensure your critique is realistic and balanced. "
            "Every claim you make is proved by citing a real paper title."
        )

        try:
            system_prompt = "You are a top-tier academic reviewer. Output clean Markdown directly starting from title `# `. NEVER output conversational greetings or AI fluff."
            report_content = self.llm.call(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.18
            )
            
            # 自动清洗掉 AI 常见的寒暄套话前缀 (如 "好的，作者。作为...编委...")
            if "# " in report_content:
                first_header_idx = report_content.find("# ")
                if first_header_idx > 0 and first_header_idx < 200:
                    report_content = report_content[first_header_idx:].strip()

            # 运行 Citation Validator 引用校验器进行审查 (过滤目标期刊名称本身)
            validated_report = self.validate_citations(
                report_content,
                aggregated_stats,
                journal_name=journal_name,
                journal_metadata=journal_metadata
            )
            logger.info("对标诊断报告生成并完成引用校验。")
            return validated_report
        except Exception as e:
            logger.error(f"生成报告异常: {str(e)}")
            raise

    def validate_citations(
        self,
        report_markdown: str,
        aggregated_stats: Dict[str, Any],
        journal_name: Optional[str] = None,
        journal_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Citation Validator (引用校验器)：
        使用归一化匹配技术，自动审查大模型生成的正文中所有被《书名号》包裹的文献。
        若发现数据库中不存在该论文，则自动追加警告标签，拒绝盲目硬替换。
        """
        valid_titles = set()
        
        # 排除目标期刊名称本身的误报
        if journal_name:
            valid_titles.add(journal_name)
        if journal_metadata and "display_name" in journal_metadata:
            valid_titles.add(journal_metadata["display_name"])
        
        # 收集所有真实存在的论文标题作为校验白名单
        all_raw_papers = (
            aggregated_stats.get("most_similar_papers", []) +
            aggregated_stats.get("recommended_references", []) +
            aggregated_stats.get("representative_novelties", [])
        )
        for p in all_raw_papers:
            if "title" in p:
                valid_titles.add(p["title"])

        def normalize_title(t: str) -> str:
            return re.sub(r"\W+", " ", t.lower()).strip()

        normalized_valid_titles = {normalize_title(t): t for t in valid_titles if t}

        # 匹配 markdown 中所有被《书名号》包裹的内容
        found_titles = re.findall(r"《(.*?)》", report_markdown)
        
        replaced_markdown = report_markdown
        for ft in set(found_titles):
            ft_norm = normalize_title(ft)
            if not ft_norm:
                continue

            # 模糊匹配：如果提取的标题长度适中，且与白名单中的任一归一化标题满足子串包含关系，则判断为通过
            matched = False
            for vt_norm, vt_orig in normalized_valid_titles.items():
                if ft_norm == vt_norm or (len(ft_norm) > 15 and (ft_norm in vt_norm or vt_norm in ft_norm)):
                    matched = True
                    break

            if not matched:
                logger.warning(f"⚠️ 校验器检测到未验证引用：《{ft}》")
                # 在书名号后插入显式警告符号，避免直接硬替换导致学术指代偏离
                replaced_markdown = replaced_markdown.replace(
                    f"《{ft}》",
                    f"《{ft}》`[⚠️ Unverified Reference]`"
                )
                
        return replaced_markdown
