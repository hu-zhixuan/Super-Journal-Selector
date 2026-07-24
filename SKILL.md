---
name: journal-profile-assistant
description: 期刊选稿画像与投稿诊断技能。基于 OpenAlex 百篇级真实论文抓取与 LLM 结构化特征提取，量化生成目标期刊的选稿偏好画像；支持论文草稿 Desk Reject 风险诊断、语义对标、多期刊投递梯队路由与模拟审稿人对话。当用户需要选择投稿期刊、评估稿件与期刊匹配度、或获取投稿前修改建议时使用本技能。
---

# 期刊选稿画像助手 (Journal Profile Assistant)

## 技能能力

| 能力 | 说明 |
|------|------|
| 选稿画像诊断 | 抓取期刊近年 100+ 篇真实论文，量化分析方法分布、理论偏好、样本量门槛、分析工具、开放科学实践、统计汇报风格 |
| 草稿对标诊断 | 传入论文草稿（.txt/.md/.docx/.pdf），语义相似度对标 Top 3 标杆论文，加权公式推荐 Top 5 应引用文献 |
| 多期刊智能路由 | 对比候选期刊，评定冲刺 / 主投 / 保底三级投递梯队 |
| Desk Reject 预警 | 审计草稿是否触发期刊方法与样本硬性死穴 |
| 引用幻觉防护 | Citation Validator 自动校验报告中引用均来自真实抓取池，未验证引用标注 [Unverified Reference] |
| 模拟审稿人对话 | 与期刊 AE 角色进行在线答辩演练 |

## 架构（4 层流水线）

```
Layer 1: OpenAlex 文献抓取（引用排序 + 关键词搜索 + Europe PMC 兜底）
Layer 2: LLM 并发结构化特征提取（Pydantic Schema 约束 + 本地缓存 + QPS 限速）
Layer 3: 纯代码统计聚合（分布/中位数/语义余弦相似度，无 LLM 幻觉）
Layer 4: LLM 循证报告生成（证据引用校验 + 投稿建议）
```

## 使用方式

### 方式一：WebUI（推荐）

```bash
python app.py
# 浏览器打开 http://127.0.0.1:7860
# Windows 可直接双击 run.bat
```

### 方式二：CLI

```bash
# 生成期刊画像报告
python main.py -j "Computers in Human Behavior" -y 3 -m 100

# 传入论文草稿进行对标诊断
python main.py -j "Strategic Management Journal" -u my_paper.docx
```

### 方式三：SDK 调用

```python
from main import run_journal_profile_skill

result = run_journal_profile_skill(
    journal="Computers in Human Behavior",
    years=3,
    max_papers=100,
    user_draft_path="my_paper.docx",  # 可选
)
print(result["report_markdown"])
```

## 环境配置

复制 `.env.example` 为 `.env` 并填写：

```ini
LLM_API_KEY=你的密钥
LLM_BASE_URL=https://fxb.supa.net.cn:6443
LLM_MODEL=deepseek-v4-flash
LLM_API_FORMAT=openai
ENABLE_LLM_DNS_PATCH=true
```

可选调优项：`LLM_EXTRACT_QPS`（提取限速，默认 4）、`LLM_EXTRACT_WORKERS`（并发线程，默认 8）、`HF_ENDPOINT`（模型镜像，默认 hf-mirror.com）、`HF_HUB_OFFLINE`（离线降级 BoW 相似度）。

## 安装依赖

```bash
pip install -r requirements.txt
```

国内用户建议预下载语义模型（避免 HuggingFace 下载卡死，约 16 秒）：

```bash
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('sentence-transformers/all-MiniLM-L6-v2')"
```

## 输出产物

- WebUI：在线报告 + 可下载 Markdown
- CLI/SDK：`output/<期刊名>/` 目录下的画像报告（Markdown）与结构化 JSON；失败样本清单 `failed_papers.json`

## 测试

```bash
python -m unittest discover -s tests
```
