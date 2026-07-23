# 期刊选稿画像助手 (Journal Profile Assistant)

百篇大样本驱动的学术期刊选稿偏好画像、多期刊智能路由与 Desk Reject 秒拒预测系统。

## 快速开始

### 1. 安装依赖

```bash
git clone https://github.com/QD8-png/-skill.git
cd --skill  # 注意目录名是 -skill
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

### 3. 运行

**方式一：WebUI（推荐）**

```bash
python app.py
```

打开浏览器访问 `http://127.0.0.1:7860`

或直接双击 `run.bat`（Windows）

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
    user_draft_path="my_paper.docx"  # 可选
)
print(result["report_markdown"])
```

### 4. 运行测试

```bash
python -m unittest discover -s tests
```

## 项目结构

```
├── app.py                  # Gradio WebUI（3个Tab：画像诊断 / 多期刊路由 / 审稿人对话）
├── main.py                 # CLI 入口 + 核心 Skill 函数
├── llm_client.py           # LLM 客户端（OpenAI/Anthropic 双格式自适应）
├── fetch_papers.py         # Layer1: OpenAlex 文献抓取（双通道+多路Fallback）
├── extract_features.py     # Layer2: LLM 并发结构化特征提取（Pydantic Schema驱动）
├── aggregate.py            # Layer3: 纯代码统计聚合 + 语义相似度对标
├── generate_profile.py     # Layer4: LLM 循证诊断报告生成 + Citation Validator
├── journal_router.py       # 多期刊智能路由（冲刺/主投/保底三级梯队）
├── journal_partitions.json # 期刊分区数据库（JCR/中科院）
├── evaluate_recommendations.py  # 推荐算法独立评估脚本
├── tests/                  # 单元测试（7个Mock测试）
├── examples/               # 示例报告
├── .env.example            # 环境变量模板
├── pyproject.toml          # 项目配置
└── requirements.txt        # 依赖清单
```

## 核心功能

| 功能 | 说明 |
|------|------|
| 选稿画像诊断 | 抓取100+篇真实论文，量化分析期刊偏好、样本门槛、理论品味 |
| 多期刊智能路由 | 自动评定冲刺/主投/保底三级投递梯队 |
| Desk Reject 预警 | 硬核审计草稿是否触发期刊物理死穴 |
| Top1 标杆解构 | 段落级逻辑链对比与像素级修改建议 |
| Citation Validator | 自动检测 AI 幻觉引用并标注警告 |
| 模拟审稿人对话 | 与 AE 角色在线答辩 |

## License

MIT
