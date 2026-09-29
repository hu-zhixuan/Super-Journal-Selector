# 期刊选稿画像助手 (Journal Profile Assistant)

抓取目标期刊近几年的论文，统计它偏好什么样的研究方法、理论、样本量和分析工具，帮你判断自己的稿子适不适合投这本期刊。

## 功能

| 功能 | 说明 |
|------|------|
| 期刊画像 | 抓取目标期刊近年约 100 篇论文，统计方法分布、常用理论、样本量分布、分析工具、开放科学实践、统计汇报方式 |
| 草稿对标 | 上传草稿（.txt/.md/.docx/.pdf），按语义相似度找出最接近的 3 篇已发表论文，并推荐 5 篇值得引用的文献 |
| 多期刊对比 | 对比几本候选期刊，给出"冲刺 / 主投 / 保底"的分档建议 |
| 风险提示 | 对照期刊的常见范式和样本量，指出草稿中可能导致直接拒稿（desk reject）的明显问题 |
| 引用校验 | 报告中引用的文献必须来自实际抓取到的论文池，校验不通过的标注为 `[Unverified Reference]` |
| 模拟审稿人 | 以期刊编辑的角色和你对话，用于提前演练审稿意见 |

**设计思路：LLM 只负责从论文中抽取信息和写报告，所有统计数字由代码计算，尽量避免 LLM 编造数据。**

```
第 1 层  OpenAlex 文献抓取（按引用排序 + 关键词搜索，Europe PMC / arXiv 兜底）
第 2 层  LLM 并发抽取结构化特征（Pydantic Schema 约束 + 本地缓存 + QPS 限速）
第 3 层  纯代码统计聚合（分布、中位数、语义余弦相似度）
第 4 层  LLM 根据统计结果生成报告，并校验引用
```

## 无密钥离线演示

不需要 API 密钥、不联网，也能跑完整个流水线并输出一份报告：

```bash
OFFLINE_DEMO=1 python main.py -j "Computers in Human Behavior" -y 3 -m 100

# Windows PowerShell
$env:OFFLINE_DEMO="1"; python main.py -j "Computers in Human Behavior"

# WebUI
OFFLINE_DEMO=1 python app.py   # 打开 http://127.0.0.1:7860
```

离线模式使用 `offline_demo/` 中预先抓取好的 OpenAlex 数据（6 篇论文及其聚合统计），不访问网络、不调用 LLM、不下载模型，结果写入 `output/<期刊名>_offline_demo/report.md`。

---

## 快速开始

### 1. 安装依赖

```bash
git clone https://github.com/hu-zhixuan/Super-Journal-Selector.git
cd Super-Journal-Selector
pip install -r requirements.txt
```

### 2. 配置 API

复制 `.env.example` 为 `.env`，填入密钥：

```ini
LLM_API_KEY=你的密钥
LLM_BASE_URL=https://fxb.supa.net.cn:6443
LLM_MODEL=deepseek-v4-flash
LLM_API_FORMAT=openai
```

可选调优项：`LLM_EXTRACT_QPS`（提取限速，默认 4）、`LLM_EXTRACT_WORKERS`（并发线程，默认 8）、`HF_ENDPOINT`（模型镜像，默认 hf-mirror.com）、`HF_HUB_OFFLINE`（离线降级 BoW 相似度）。

### 3. 运行

**方式一：WebUI（推荐）**

```bash
python app.py
```

打开浏览器访问 `http://127.0.0.1:7860`；或直接双击 `run.bat`（Windows）。

**方式二：命令行**

```bash
# 生成期刊画像报告
python main.py -j "Computers in Human Behavior" -y 3 -m 100

# 传入论文草稿进行对标诊断
python main.py -j "Strategic Management Journal" -u my_paper.docx
```

**方式三：Python SDK**

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

### 4. 运行测试

```bash
python -m unittest discover -s tests
# 或
pytest --cov=. tests/
```

---

## 项目结构

```
├── app.py                  # Gradio WebUI（3个Tab：画像诊断 / 多期刊路由 / 审稿人对话）
├── main.py                 # CLI 入口 + 核心 Skill 函数
├── network_config.py       # 网络引导（HF 镜像/离线模式，须最先导入）
├── llm_client.py           # LLM 客户端（OpenAI/Anthropic 双格式自适应）
├── fetch_papers.py         # 第1层: OpenAlex 文献抓取（多数据源兜底 + 精确期刊匹配）
├── extract_features.py     # 第2层: LLM 并发结构化特征抽取
├── aggregate.py            # 第3层: 纯代码统计聚合 + 语义相似度
├── generate_profile.py     # 第4层: LLM 报告生成 + 引用校验
├── journal_router.py       # 多期刊对比（冲刺/主投/保底分档）
├── journal_partitions.json # 期刊分区数据库（JCR/中科院）
├── evaluate_recommendations.py  # 推荐打分的小规模评估脚本
├── tests/                  # 单元测试
├── examples/               # 示例报告
├── .env.example            # 环境变量模板
├── pyproject.toml          # 项目配置（含 CI 的 ruff/mypy/pytest）
└── requirements.txt        # 依赖清单
```

---

## 常见问题

**Q: 进度卡在"正在启动统计引擎计算余弦相似度... 80%"不动？**

原因：首次运行需下载语义模型 `all-MiniLM-L6-v2`（约 90MB），国内直连 huggingface.co 速度较慢时会反复重试。

解决（三级保障，已内置 `network_config.py` 自动处理）：

1. **ModelScope 国内源预下载（推荐）**：
   ```bash
   pip install modelscope
   python -c "from modelscope import snapshot_download; snapshot_download('sentence-transformers/all-MiniLM-L6-v2')"
   ```
   下载后程序会自动使用本地缓存，之后加载不再联网。
2. **hf-mirror 镜像**：未预下载时，默认走 hf-mirror.com 镜像自动下载。
3. **完全离线降级**：在 `.env` 中设置 `HF_HUB_OFFLINE=1`，自动降级为 BoW 词频相似度（无需任何下载，精度略降）。

也可用环境变量 `EMBEDDING_MODEL_PATH` 显式指定本地模型目录。

**Q: `LLM_API_KEY` 已配置但调用 401/403？**

检查密钥是否有效、`LLM_BASE_URL` 是否正确；本工具自动带备用端点（DeepSeek 官方）兜底，仍需主端点可用。

**Q: 期刊名相似会匹配错吗？**

已实现"精确匹配优先"：如 `Computers in Human Behavior` 优先命中主刊，不会被同名子刊 `Computers in Human Behavior Reports` 挤掉。

---

## 局限性

- **"直接拒稿风险"不是经过验证的预测模型。** 它是 LLM 在看过期刊统计画像后给出的判断，没有用真实的投稿/拒稿记录做过检验，只能当作检查清单参考。
- **推荐评估规模很小。** `evaluate_recommendations.py` 用的是 10 篇人工构造的论文和手工标注的答案，只能说明打分逻辑能跑通、比随机好，不代表在真实期刊上的效果。
- **画像受抓取样本影响。** 默认按引用数排序抓取，高被引论文偏老、偏经典，未必代表期刊近期的偏好；OpenAlex 的摘要和元数据也有缺失。
- **特征抽取依赖 LLM。** 方法、样本量等字段由 LLM 从摘要中抽取，摘要没写清楚时会抽错或留空，统计结果会跟着偏。
- **分区数据需要手动维护。** `journal_partitions.json` 是静态文件，JCR / 中科院分区每年更新，需要自己同步。

## License

MIT
