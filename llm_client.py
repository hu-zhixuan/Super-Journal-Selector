import os
import json
import re
import time
import random
import logging
import requests
import urllib3
import urllib3.util.connection as urllib3_cn
from typing import Optional, Dict, Any
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

# ==================== Socket 层 DNS 劫持补丁 ====================
ENABLE_LLM_DNS_PATCH = os.getenv("ENABLE_LLM_DNS_PATCH", "false").lower() == "true"
DEFAULT_DIRECT_IP = os.getenv("LLM_DIRECT_IP", "114.80.15.146")

def patched_create_connection(address, *args, **kwargs):
    host, port = address
    if host == "fxb.supa.net.cn" and DEFAULT_DIRECT_IP:
        # 尝试强制导向物理 IP，如果失败则回退至标准 DNS 域名解析
        try:
            return urllib3_cn._orig_create_connection((DEFAULT_DIRECT_IP, port), *args, **kwargs)
        except Exception:
            pass
    return urllib3_cn._orig_create_connection(address, *args, **kwargs)

def install_dns_patch():
    """
    显式安装 Socket 层 DNS 直连补丁。
    默认由环境变量 ENABLE_LLM_DNS_PATCH 控制。
    """
    if ENABLE_LLM_DNS_PATCH:
        if not hasattr(urllib3_cn, "_orig_create_connection"):
            urllib3_cn._orig_create_connection = urllib3_cn.create_connection
            urllib3_cn.create_connection = patched_create_connection
            logger.info(f"已成功加载 Socket DNS 直连补丁：fxb.supa.net.cn -> {DEFAULT_DIRECT_IP}")
    else:
        logger.info("Socket DNS 直连补丁处于关闭状态（按需开启）")

# 执行初始化补丁检查
install_dns_patch()
# ===============================================================


# ==================== Prompt 与算法版本控制 ====================
import hashlib

EXTRACTION_PROMPT_VERSION = "v1.3"
REPORT_PROMPT_VERSION = "v2.0"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def get_prompt_fingerprint(prompt_version: str, model_name: str, temperature: float, system_prompt: str) -> str:
    """
    基于 Prompt 版本、模型名、温度和系统 Prompt 串生成 10 位指纹，防止缓存被旧 Prompt 污染。
    """
    raw_str = f"{prompt_version}:{model_name}:{temperature}:{system_prompt}"
    return hashlib.md5(raw_str.encode("utf-8")).hexdigest()[:10]
# ===============================================================


class LLMClient:
    """
    统一封装的 LLM 客户端，适配 Anthropic Messages API 格式 (/v1/messages)。
    支持 requests.Session 连接池复用、超时及最大 token 可配置、JSON 解析重试修复。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 60,
        max_tokens: int = 4000
    ):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "https://fxb.supa.net.cn:6443")
        self.model = model or os.getenv("LLM_MODEL", "deepseek-v4-flash")
        self.timeout = timeout
        self.max_tokens = max_tokens

        # Token 与成本度量统计
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_api_calls = 0
        self.token_source = "api"  # "api" 或 "estimated"

        # 模型单价映射表 (输入 / 输出每百万 Token 费率，以美元计)
        self.MODEL_PRICING = {
            "deepseek-v4-flash": {
                "input_per_1m_tokens": 0.1,
                "output_per_1m_tokens": 0.2,
            },
            "deepseek-v3": {
                "input_per_1m_tokens": 0.14,
                "output_per_1m_tokens": 0.28,
            }
        }

        # 规范化 Base URL 路径
        self.url = self.base_url.rstrip("/")
        if not self.url.endswith("/v1/messages"):
            self.url = f"{self.url}/v1/messages"

        if not self.api_key or self.api_key == "your_api_key_here":
            logger.warning("未检测到有效的 LLM_API_KEY。若调用 API 将导致鉴权失败，请在 .env 中正确配置。")

        # 使用 Session 连接池复用，提升批量请求效率
        self.session = requests.Session()
        self.session.trust_env = False
        # 强行忽略系统代理环境变量
        self.session.proxies = {
            "http": None,
            "https": None
        }

    def call(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful academic research assistant.",
        temperature: float = 0.3,
        max_retries: int = 5,
    ) -> str:
        """
        发起 Anthropic Messages 协议请求
        """
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        for attempt in range(1, max_retries + 1):
            try:
                # 针对代理端点非标 6443 端口与连接池陈旧连接进行 SSL 安全容错与自动刷新
                response = self.session.post(
                    self.url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                    verify=False
                )
                response.raise_for_status()
                
                resp_data = response.json()
                
                # 警告输出截断
                if resp_data.get("stop_reason") == "max_tokens":
                    logger.warning("LLM 输出被 max_tokens 截断，可能导致后续 JSON 或文本内容解析不全。")

                content_list = resp_data.get("content", [])
                text_parts = [
                    block.get("text", "")
                    for block in content_list
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                text = "".join(text_parts).strip()
                if not text:
                    logger.error(f"API 响应结构异常，文本内容为空: {resp_data}")
                    raise ValueError("Anthropic API 返回内容为空")

                # 解析统计使用量数据
                usage = resp_data.get("usage", {})
                p_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
                c_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
                self.total_api_calls += 1
                if p_tokens or c_tokens:
                    self.total_prompt_tokens += p_tokens
                    self.total_completion_tokens += c_tokens
                else:
                    # 兜底粗估，并标记来源
                    self.token_source = "estimated"
                    self.total_prompt_tokens += len(prompt.split()) + len(system_prompt.split())
                    self.total_completion_tokens += len(text.split())

                return text

            except Exception as e:
                # 若遇到 SSL EOF 协议中断或 TCP 连接池陈旧断开，自动重建 Session 清除脏连接
                if "SSL" in str(e) or "Connection" in str(e):
                    logger.warning(f"检测到 SSL/TCP 连接池异常 ({str(e)})，正在自动重建 Session 刷新连接...")
                    self.session = requests.Session()

                is_rate_limit = False
                wait_time = 0
                
                # 检查是否为 HTTP 429 速率限制错误，并读取 Retry-After 头
                if hasattr(e, "response") and e.response is not None:
                    if getattr(e.response, "status_code", None) == 429:
                        is_rate_limit = True
                        retry_after = e.response.headers.get("Retry-After")
                        if retry_after:
                            try:
                                wait_time = int(retry_after)
                            except ValueError:
                                pass

                if is_rate_limit:
                    logger.warning(f"触发 API 频率限制 (HTTP 429) (尝试 {attempt}/{max_retries})，正在执行退避重试...")
                else:
                    logger.warning(f"LLM API 接入调用失败 (尝试 {attempt}/{max_retries}): {str(e)}")

                if attempt == max_retries:
                    raise RuntimeError(f"All LLM API retry attempts failed. Last error: {e}") from e

                # 避让逻辑
                sleep_time = wait_time if wait_time > 0 else (2 ** attempt) + random.uniform(0.5, 2.0)
                if is_rate_limit and wait_time == 0:
                    sleep_time += 3.0  # 针对频率限制额外延长等待时间
                
                logger.info(f"等待 {sleep_time:.2f} 秒后重试...")
                time.sleep(sleep_time)

        raise RuntimeError("All LLM API retry attempts failed")

    def extract_json_from_text(self, raw_output: str) -> Dict[str, Any]:
        """
        强力提取文本中的 JSON 部分，支持 markdown、object 和 array。
        """
        candidates = []

        # 1. 优先提取 markdown 代码块
        code_block = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_output, re.DOTALL)
        if code_block:
            candidates.append(code_block.group(1))

        # 2. 匹配可能的对象 `{...}` 贪婪与非贪婪最大区间
        obj_start = raw_output.find("{")
        obj_end = raw_output.rfind("}")
        if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
            candidates.append(raw_output[obj_start:obj_end + 1])

        # 3. 匹配可能的数组 `[...]` 最大区间
        arr_start = raw_output.find("[")
        arr_end = raw_output.rfind("]")
        if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
            candidates.append(raw_output[arr_start:arr_end + 1])

        # 4. 尝试直接解析全文
        candidates.append(raw_output)

        for candidate in candidates:
            try:
                candidate_str = candidate.strip()
                if candidate_str:
                    return json.loads(candidate_str)
            except json.JSONDecodeError:
                continue

        raise ValueError("LLM 返回内容中不包含任何合法的 JSON 结构")

    def call_json(
        self,
        prompt: str,
        system_prompt: str = "You are an AI research analyst. You must output valid JSON only. Do not wrap in markdown blocks.",
        temperature: float = 0.1,
        max_retries: int = 5,
    ) -> Dict[str, Any]:
        """
        请求并强力解析 JSON 输出，内置 JSON 损坏自动重试修复机制。
        """
        raw_output = self.call(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_retries=max_retries,
        )

        try:
            return self.extract_json_from_text(raw_output)
        except ValueError as e:
            logger.warning(f"首次 JSON 解析失败: {e}。触发大模型自我修复调用...")
            repair_prompt = f"""
你上一次返回的输出内容不符合合法的 JSON 格式。
错误解析提示: {str(e)}

请仔细修改，只返回合法的 JSON 对象或数组。不要输出任何解释文字，也不要使用 markdown 语法包裹。
你上一次返回的原始输出如下:
{raw_output}
"""
            try:
                repaired_output = self.call(
                    prompt=repair_prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_retries=max_retries,
                )
                return self.extract_json_from_text(repaired_output)
            except Exception as e_repair:
                logger.error(f"大模型自我修复 JSON 失败: {e_repair}")
                raise ValueError(f"大模型自我修复 JSON 失败: {e_repair}") from e_repair

    def get_cost_statistics(self) -> Dict[str, Any]:
        """
        获取当前的 API 调用次数、Token 消耗及预估费用。
        """
        pricing = self.MODEL_PRICING.get(self.model)
        cost_usd = None
        if pricing:
            in_rate = pricing.get("input_per_1m_tokens", 0.1)
            out_rate = pricing.get("output_per_1m_tokens", 0.2)
            cost_usd = round(
                (self.total_prompt_tokens / 1_000_000.0) * in_rate +
                (self.total_completion_tokens / 1_000_000.0) * out_rate,
                6
            )
        return {
            "total_api_calls": self.total_api_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "token_source": self.token_source,
            "estimated_cost_usd": cost_usd,
            "estimated_cost_cny": round(cost_usd * 7.2, 5) if cost_usd is not None else None
        }

