import os
import json
import logging
from typing import Dict, Any, Optional
from llm_client import LLMClient

logger = logging.getLogger(__name__)


class ProfileGenerator:
    """
    层④：战略生成层。将层③输出的量化统计特征，与用户待投稿文本/目标进行高段位提示词工程拼装，
    利用 LLM 的高级推理，生成极度精准、冷酷揭露隐性潜规则、刺破修稿痛点的硬核学术诊断与手术指导报告。
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
        基于统计聚合指标与可选的用户草稿文本，生成极具穿透力与实操价值的 Markdown 画像与修稿策略
        """
        logger.info(f"正在为期刊 '{journal_name}' 撰写深度硬核偏好画像与手术级策略报告...")

        stats_json = json.dumps(aggregated_stats, ensure_ascii=False, indent=2)

        user_draft_section = ""
        if user_draft_text and user_draft_text.strip():
            user_draft_section = f"""
### 【核心对标目标：用户拟投稿论文摘要/大纲冷酷诊断与操刀】
用户的论文文本（摘要/大纲/草稿）如下：
```text
{user_draft_text[:3000]}
```
请在第三部分，以顶级主编的挑剔眼光直接撕掉这段文字里的“学生气”、“普通水文感”和“假大空套话”，给出字面级的手术式对比与重构！
"""
        else:
            user_draft_section = """
（注：当前用户未提供具体投稿摘要草稿。在第三部分请直接列出该刊极具实战杀伤力的“高通关率 Abstract 黄金句式模板”以及核心论证补强手术方案。）
"""

        prompt = f"""
你是一名在顶尖学术顶刊（如 {journal_name}）担任多年主编（Editor-in-Chief）及铁面无私的高级审稿人。
现在我们通过底层数据工程将该刊最近发表论文的量化统计聚合指标精准提取了出来，真实统计数据如下：

```json
{stats_json}
```

{user_draft_section}

请务必根据以上真实的量化聚合数据，为准备向 `{journal_name}` 投稿的科研学者输出一份**极度老辣、精准刺破痛点、一针见血、冷酷揭露主编选稿潜规则**的《期刊深度偏好画像与手术级修稿报告》。

### 【写作口吻与基调硬性约束 — 必须严格执行！】
1. **彻底拒绝废话与套话**：严禁出现“非常优秀的研究”、“建议进一步优化和加强”等温吞水废话！每一句话必须像外科手术刀一样精准利落，直戳学者的认知盲区。
2. **冷酷揭露隐性潜规则**：站在主编初审筛稿（Desk Reject）的第一视角，用客观冷酷的口吻指出：这本刊到底挑什么口音的黑话、看不起什么层次的方法学偷懒、最反感哪类写作套路。
3. **实操可落地（拒绝抽象指导）**：修稿策略部分绝对不能只给“方向性建议”，必须直接给出**改写前后对比方案、强穿透力的逻辑重构话术、以及审稿人无可挑剔的实证防御框架**！

报告必须严格按照以下排版格式与专业模块生成：

# 《{journal_name}》底层选稿品味剖析与手术级投稿指南

## 一、 主编初审视角下的选稿底层逻辑与偏好画像
1. **范式冷酷分层与被引现实**：用统计数据说话！精准解剖各方法学（实证、定性、混研、计算AI）的实际发文门槛和均次被引对决。揭露该刊目前把流量和优质资源（高被引）重点倾斜给了哪类新锐赛道，哪类老旧范式正在边缘化。
2. **理论“对黑话”与对话场域**：直接点出本刊最高频的核心理论（Top Theories）。严厉提醒作者：如果引言部分不跟这些核心构念与理论对话，主编第一眼就会觉得“根本不具有本刊的学术血统，属于外行投错门”。
3. **数据量级与计量工具硬门槛**：明确指出实证研究在样本量（Min/Median）以及工具链上的底层“资格线”。直接告诉作者低于什么样本量或拿什么过时的简单回归来投，会被审稿人直接以“Methodological Rigor 不及格”秒杀。
4. **近期爆款文章“破圈”基因解密**：分析最近高被引代表作为什么能火，提炼该刊核心编委圈当前最渴求的研究品味与选题亮点。

## 二、 审稿雷区与“一击必杀”初审拒稿清单 (Desk Reject Kill Switches)
直接指出 3~4 个触犯必死的“初审拒稿红线”。例如：
- 伪造理论空白（如写“填补了目前研究空白”等主编最憎恨的废话）
- 方法学偷懒（如单一横截面数据且样本量不及格却试图谈因果）
- 理论与实证两张皮（只罗列显著性，缺乏对行为/机制本质的底层剖析）
*(必须针对上面聚合出来的低频数据和短板，句句见血地揭露！)*

## 三、 手术级投稿对标诊断与重构操刀方案 (Surgical Tailoring Blueprint)
{"请直接针对上面用户提供的待投稿摘要进行残酷诊断与全方位的重构改造！" if user_draft_text else "给出直接可套用、经得起审稿人极其严苛挑剔的通关实战改造模板："}
1. **标题切除与重塑 (Title Surgery)**：指出原标题/传统普通标题为什么平庸无奇。直接给出 **1~2 个充满顶刊理论张力、带因果机制穿透力的重构标题对照**！
2. **Abstract 七步逻辑重写模板 (Abstract Reframing)**：严厉纠正平铺直叙式写法。手把手展示如何写出让主编一眼无法拒绝的黄金摘要逻辑链路（痛点钩子 -> 理论冲突 -> 实证设计 -> 核心反直觉发现 -> 理论本质颠覆）。
3. **实证防御与审稿人口密闭环 (Methodological Defense)**：为了防止外审时被极其挑刺的审稿人（Reviewer #2）打回，现在立刻需要在论文方法学或稳健性检验（Robustness Check）里额外补做哪 2 个关键检验或数据说明。

排版请严格使用高清晰度的 Markdown 和表格/引用区块，确保这份报告不仅是一份分析，更是帮学者从“被拒稿边缘”直接拉到“外审修回大胜”的救命指南！
"""

        system_prompt = (
            "You are a legendary Editor-in-Chief and elite peer reviewer at top-tier journals. "
            "Your feedback is sharp, surgical, diagnostic, and completely void of fluff or academic politeness. "
            "You pierce right through authors' pain points and deliver game-changing revision strategies in authoritative Chinese."
        )

        try:
            report_content = self.llm.call(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.45,
                max_retries=3,
            )
            logger.info("深度硬核偏好画像与手术级修稿报告生成完毕。")
            return report_content
        except Exception as e:
            logger.error(f"生成最终硬核报告异常: {str(e)}")
            raise
