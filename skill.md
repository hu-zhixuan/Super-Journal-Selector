---
name: journal-profile-assistant
description: 百篇大样本驱动的学术期刊选稿偏好画像、全网多期刊智能路由与主编秒拒死穴预测 Agent Skill
version: 2.5.0
author: AI4SS Team
tags:
  - AI4SS
  - AI for Social Science
  - Academic Publishing
  - Journal Profile
  - Peer Review
---

# 📊 期刊选稿画像助手 (Journal Profile Assistant)
> **书生·浦砚 (Intern Discovery) 大赛自由赛道 (AI4SS) 参赛作品**

`journal-profile-assistant` 是一个针对学术论文投稿“最后一公里”痛点打造的 **百篇大样本驱动的期刊选稿底层偏好画像与全网智能路由 Agent Skill**。

通过结合 **OpenAlex 开放学术大样本 (100+ 篇)**、**DeepSeek-v4-flash 多线程结构化抽取** 与 **纯代码余弦聚类算法**，解决传统选稿工具“泛泛而谈、缺乏特异性指向、AI 幻觉编造”的致命缺陷。

---

## 🌟 核心杀手锏功能

1. **🧭 全网多期刊梯队智能路由大盘 (Smart Multi-Journal Router)**：
   输入草稿即可自动比对全网学科期刊池，输出包含契合度得分与录用估计的 **冲刺 (Reaching)**、**主投 (Target)**、**保底 (Safe)** 三级投递阵列。

2. **🚨 主编 Desk Reject (秒拒) 致命死穴预测 (Desk Reject Predictor)**：
   硬核审计草稿是否触发该期刊近 3 年大样本的物理死穴（如范式严重偏向非主流、样本量未达中位数线）。

3. **🔬 Top 1 标杆文献段落级逻辑链解构 (Paragraph-Level Logical Alignment)**：
   一比一拆解该期刊近期最相似高引标杆论文的引言与方法论写作骨架，提供像素级对齐修改建议。

4. **⚖️ 归一化 Citation Validator 真实性校验**：
   正文引用的文献自动与真实抓取池比对，未验证文献显示 `[⚠️ Unverified Reference]` 显式警告，彻底杜绝 AI 虚假引用。

5. **📊 实时 Token 成本与耗时监控**：
   单次百篇分析仅消耗约 2.3 万 Tokens（预估费用低至 0.02 元人民币），耗时支持秒级推流展示。

---

## 🛠️ 输入与输出规范 (Skill Specification)

### 1. 输入参数 (Input Schema)
```json
{
  "journal_name": "Computers in Human Behavior",
  "years": 3,
  "max_papers": 100,
  "user_draft_text": "Optionally pass title/abstract or path to .docx/.pdf file"
}
```

### 2. 输出结构 (Output Schema)
```json
{
  "status": "success",
  "journal_metadata": {
    "display_name": "Computers in Human Behavior",
    "issn": "0747-5632",
    "jcr_zone": "Q1",
    "cas_zone": "1区",
    "is_top": "是 (Top 期刊)"
  },
  "aggregated_stats": {
    "total_papers_analyzed": 100,
    "method_distribution": {"Quantitative_Empirical": {"count": 62, "percentage": 62.0}},
    "sample_size_stats": {"min": 312, "median": 761, "max": 4500},
    "most_similar_papers": [],
    "recommended_references": []
  },
  "cost_statistics": {
    "total_api_calls": 17,
    "total_prompt_tokens": 17918,
    "total_completion_tokens": 5927,
    "estimated_cost_cny": 0.0214,
    "elapsed_seconds": 44.19
  },
  "report_markdown": "# 《Computers in Human Behavior》选稿偏好与循证对标诊断报告..."
}
```

---

## 🚀 快速使用说明 (Quick Start)

### 1. 命令行调用 (SDK / Python Interface)
```python
from main import run_journal_profile_skill

# 一键运行 Skill
result = run_journal_profile_skill(
    journal="Computers in Human Behavior",
    years=3,
    max_papers=100,
    user_draft="examples/my_draft_test.docx"
)

print("处理状态:", result["status"])
print("报告保存位置:", result["report_markdown"])
```

### 2. WebUI 界面启动
```bash
python app.py
```
访问 `http://127.0.0.1:7860` 即可在可视化界面中使用。

### 3. 运行自动化测试套件 (0.019 秒全量 Mock 通过)
```bash
python -m unittest discover -s tests
```

---

## 📄 许可与竞赛备案
* **参赛赛道**：书生·浦砚 AI4SS (AI for Social Science) 自由赛道
* **开源协议**：[MIT License](LICENSE)
