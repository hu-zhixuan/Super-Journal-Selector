---
name: journal-profile-assistant
description: 期刊选稿画像与投稿诊断技能。基于 OpenAlex 百篇级真实论文抓取与 LLM 结构化特征提取，量化生成目标期刊的选稿偏好画像；支持论文草稿 Desk Reject 风险诊断、语义对标、多期刊投递梯队路由与模拟审稿人对话。当用户需要选择投稿期刊、评估稿件与期刊匹配度、或获取投稿前修改建议时使用本技能。
when_to_use: 当用户询问投稿期刊选择、期刊选稿偏好画像、稿件与期刊匹配度评估、Desk Reject 风险诊断或投稿前修改建议时，调用本技能生成基于真实大样本数据的循证报告。
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

## 执行步骤（Agent 调用引导）

1. **环境校验**：确认项目根目录为 `提交包/-skill/`，存在 `.env`（含 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL`），依赖已按 `requirements.txt` 安装。缺 `.env` 时提示用户复制 `.env.example` 配置后重试。
2. **确认输入**：目标期刊英文全称（必填，如 `Computers in Human Behavior`）；可选论文草稿文件路径（.txt/.md/.docx/.pdf）；样本年份 `years`（默认 3）、样本量 `max_papers`（默认 100）。
3. **执行流水线**：
   - CLI：`python main.py -j "<期刊名>" -y <年份> -m <数量> [-u <草稿路径>] [-o <输出路径>]`
   - SDK：`from main import run_journal_profile_skill; result = run_journal_profile_skill(journal=..., years=..., max_papers=..., user_draft_path=...)`
4. **读取产物**：
   - 报告 Markdown：`output/<期刊名>/report.md`（未提供草稿）或 `output/<期刊名>_with_draft_<hash>/report.md`（提供草稿）
   - 中间数据：`papers.json` / `features.json` / `aggregated_stats.json` / `execution_stats.json`；失败样本 `failed_papers.json`
5. **失败处理**：返回 `status: "error"` 时，按 `error_code` 处理——
   - `NO_PAPERS_FETCHED`：期刊名无法在 OpenAlex 定位或无带摘要样本，换用其他期刊名或加大 `years`
   - `FEATURE_EXTRACTION_FAILED`：LLM 提取 0 篇成功，检查 `.env` 密钥/额度与 `LLM_EXTRACT_QPS`
   - 其他异常：按 `message` 排查网络 / 依赖 / 模型缓存（详见"常见问题"）

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

## 常见问题

**Q: 进度卡在"正在启动统计引擎计算余弦相似度"不动？**

原因：首次运行需下载语义模型 `all-MiniLM-L6-v2`（约 90MB），国内直连 huggingface.co 慢时会反复重试。
解决：ModelScope 预下载（见"安装依赖"）、或 `.env` 设 `HF_HUB_OFFLINE=1` 降级为 BoW 相似度、或用 `EMBEDDING_MODEL_PATH` 指定本地模型目录。