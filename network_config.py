"""
网络引导配置（必须在所有第三方库之前最先导入）。

背景：国内网络直连 huggingface.co 常被阻断，sentence-transformers 首次
下载 all-MiniLM-L6-v2 模型时会出现无限重试（WinError 10060），
导致流水线长时间卡死在"统计引擎计算余弦相似度"阶段。

注意：huggingface_hub 的 ENDPOINT 等常量在模块导入时即固化，
且 gradio 等库会间接提前导入 huggingface_hub，
因此 HF_ENDPOINT 必须在任何第三方库导入之前设置才可靠生效。

可选环境变量（均可写入 .env）：
- HF_ENDPOINT       覆盖默认镜像地址（海外用户可设回 https://huggingface.co）
- HF_HUB_OFFLINE=1  完全离线模式：仅使用本地模型缓存；
                    缓存缺失时快速失败并自动降级为 BoW 词频相似度，
                    不再产生任何网络重试等待。
"""
import os

# 先加载 .env，使用户自定义的 HF_ENDPOINT / HF_HUB_OFFLINE 优先生效
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 强制清理系统代理环境变量，确保物理直连，
# 防止本地 VPN/Clash 对镜像站造成 SSL 中间人干扰（EOF in violation of protocol）。
# 与 llm_client.py 中的代理清理策略保持一致，此处上移以覆盖 HF 模型下载链路。
for _proxy_var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                   "http_proxy", "https_proxy", "all_proxy"]:
    os.environ.pop(_proxy_var, None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

# 国内默认走 hf-mirror 镜像；用户已通过环境变量或 .env 配置时尊重用户设置
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def resolve_embedding_model_path(model_name: str = "all-MiniLM-L6-v2") -> str:
    """
    解析 sentence-transformers 模型的实际加载路径，本地缓存优先，
    避免每次启动都向 huggingface 发起校验/下载请求。

    优先级：
    1. 环境变量 EMBEDDING_MODEL_PATH 显式指定的本地目录
    2. ModelScope 国内镜像缓存（预下载方式见 README 常见问题，国内约 16 秒下完）
    3. HuggingFace Hub 模型名（本模块已将 HF_ENDPOINT 默认指向 hf-mirror 镜像）
    """
    custom_path = os.getenv("EMBEDDING_MODEL_PATH")
    if custom_path and os.path.isdir(custom_path):
        return custom_path

    modelscope_cache = os.path.join(
        os.path.expanduser("~"), ".cache", "modelscope", "models",
        f"sentence-transformers--{model_name}", "snapshots", "master",
    )
    if os.path.isfile(os.path.join(modelscope_cache, "modules.json")):
        return modelscope_cache

    return model_name
