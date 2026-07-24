import json
import logging
import os
import random
import re
import time
from typing import Any, Dict, Optional

import requests
import urllib3
import urllib3.util.connection as urllib3_cn
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

# 清理代理环境变量，确保网络直连，防止本地代理工具干扰
for proxy_var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    os.environ.pop(proxy_var, None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

# ==================== Socket 层 DNS 直连优化 ====================
ENABLE_LLM_DNS_PATCH = os.getenv("ENABLE_LLM_DNS_PATCH", "true").lower() == "true"
DEFAULT_DIRECT_IP = os.getenv("LLM_DIRECT_IP", "114.80.15.146")


def patched_create_connection(address, *args, **kwargs):
    host, port = address
    if host == "fxb.supa.net.cn" and DEFAULT_DIRECT_IP:
        try:
            return urllib3_cn._orig_create_connection((DEFAULT_DIRECT_IP, port), *args, **kwargs)
        except Exception:
            pass
    return urllib3_cn._orig_create_connection(address, *args, **kwargs)


def install_dns_patch():
    if ENABLE_LLM_DNS_PATCH:
        if not hasattr(urllib3_cn, "_orig_create_connection"):
            urllib3_cn._orig_create_connection = urllib3_cn.create_connection
            urllib3_cn.create_connection = patched_create_connection
            logger.info(f"已成功加载 Socket DNS 直连补丁：fxb.supa.net.cn -> {DEFAULT_DIRECT_IP}")
    else:
        logger.info("Socket DNS 直连补丁处于关闭状态（按需开启）")


install_dns_patch()
# ===============================================================


# ==================== Prompt 与算法版本控制 ====================
import hashlib  # noqa: E402  # 分组注释块后的模块导入

EXTRACTION_PROMPT_VERSION = "v1.3"
REPORT_PROMPT_VERSION = "v2.0"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def get_prompt_fingerprint(prompt_version: str, model_name: str, temperature: float, system_prompt: str) -> str:
    raw_str = f"{prompt_version}:{model_name}:{temperature}:{system_prompt}"
    return hashlib.md5(raw_str.encode("utf-8")).hexdigest()[:10]


# ===============================================================


class LLMClient:
    """
    统一封装的 LLM 客户端，自动适配 OpenAI / Anthropic 两种 API 格式。
    默认使用 OpenAI Chat Completions 格式（适配大多数 LLM 服务端），
    可通过 LLM_API_FORMAT=anthropic 切换为 Anthropic Messages 格式。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ):
        self.api_key: Optional[str] = api_key or os.getenv("LLM_API_KEY")
        self.base_url: str = base_url or os.getenv("LLM_BASE_URL") or "https://fxb.supa.net.cn:6443"
        self.model: str = model or os.getenv("LLM_MODEL") or "deepseek-v4-flash"
        # 可用 LLM_TIMEOUT / LLM_MAX_TOKENS 环境变量全局调整（长报告生成在调用处单独放宽）
        self.timeout = timeout if timeout is not None else int(os.getenv("LLM_TIMEOUT", "60"))
        self.max_tokens = max_tokens if max_tokens is not None else int(os.getenv("LLM_MAX_TOKENS", "4000"))
        self.fallback_api_key = os.getenv("LLM_FALLBACK_API_KEY", "")

        # API 格式: "openai" (默认) 或 "anthropic"
        self.api_format = os.getenv("LLM_API_FORMAT", "openai").lower()

        # Token 与成本度量统计
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_api_calls = 0
        self.token_source = "api"

        # 模型单价映射表
        self.MODEL_PRICING = {
            "deepseek-v4-flash": {
                "input_per_1m_tokens": 0.1,
                "output_per_1m_tokens": 0.2,
            },
            "deepseek-v3": {
                "input_per_1m_tokens": 0.14,
                "output_per_1m_tokens": 0.28,
            },
            "deepseek-chat": {
                "input_per_1m_tokens": 0.14,
                "output_per_1m_tokens": 0.28,
            },
            "minimax-2.7": {
                "input_per_1m_tokens": 0.1,
                "output_per_1m_tokens": 0.2,
            },
        }

        # 规范化 URL
        self.url = self._build_url(self.base_url)

        if not self.api_key or self.api_key == "your_api_key_here":
            logger.warning("未检测到有效的 LLM_API_KEY。若调用 API 将导致鉴权失败，请在 .env 中正确配置。")

        self._init_session()

    def _build_url(self, base: str) -> str:
        """根据 API 格式构建请求 URL"""
        url = base.rstrip("/")
        if self.api_format == "anthropic":
            if not url.endswith("/v1/messages"):
                # 如果已含 /v1 但不是 /v1/messages，替换；否则追加
                if url.endswith("/v1"):
                    url = url + "/messages"
                elif "/v1/" in url:
                    pass  # 已有自定义路径，不动
                else:
                    url = url + "/v1/messages"
        else:
            # OpenAI 格式
            if not url.endswith("/v1/chat/completions"):
                if url.endswith("/v1"):
                    url = url + "/chat/completions"
                elif "/v1/chat/completions" in url:
                    pass
                elif "/v1/" in url:
                    # 去掉旧的 /v1/xxx，替换为 /v1/chat/completions
                    url = url[: url.index("/v1/")] + "/v1/chat/completions"
                else:
                    url = url + "/v1/chat/completions"
        return url

    def _init_session(self):
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.proxies = {"http": None, "https": None}

    def _build_headers(self, api_key: str) -> Dict[str, str]:
        """根据 API 格式构建请求头"""
        if self.api_format == "anthropic":
            return {"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        else:
            return {"Authorization": f"Bearer {api_key}", "content-type": "application/json"}

    def _build_payload(
        self, prompt: str, system_prompt: str, temperature: float, max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """根据 API 格式构建请求体"""
        mt = max_tokens or self.max_tokens
        if self.api_format == "anthropic":
            return {
                "model": self.model,
                "max_tokens": mt,
                "temperature": temperature,
                "system": system_prompt,
                "messages": [{"role": "user", "content": prompt}],
            }
        else:
            return {
                "model": self.model,
                "max_tokens": mt,
                "temperature": temperature,
                "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            }

    def _parse_response(self, resp_data: Dict[str, Any]) -> str:
        """根据 API 格式解析响应文本"""
        if self.api_format == "anthropic":
            content_list = resp_data.get("content", [])
            text_parts = [
                block.get("text", "")
                for block in content_list
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            return "".join(text_parts).strip()
        else:
            # OpenAI format
            choices = resp_data.get("choices", [])
            if choices:
                if choices[0].get("finish_reason") == "length":
                    logger.warning("LLM 输出达到 max_tokens 上限被截断，可能导致 JSON 不完整。")
                message = choices[0].get("message", {})
                return (message.get("content") or "").strip()
            return ""

    def _parse_usage(self, resp_data: Dict[str, Any]) -> tuple:
        """解析 token 使用量，返回 (prompt_tokens, completion_tokens)"""
        usage = resp_data.get("usage", {})
        if self.api_format == "anthropic":
            return (usage.get("input_tokens", 0), usage.get("output_tokens", 0))
        else:
            return (usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))

    def call(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful academic research assistant.",
        temperature: float = 0.3,
        max_retries: int = 5,
        timeout: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """发起 LLM API 请求（自动适配 OpenAI / Anthropic 格式）"""
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        # 构建端点列表
        endpoints = []
        endpoints.append((self.url, self.api_key))
        # 如果主端点是 fxb 服务，添加 DeepSeek 官方作为备用
        if "fxb.supa.net.cn" in self.url:
            fallback_key = self.fallback_api_key or self.api_key
            if self.api_format == "anthropic":
                endpoints.append(("https://api.deepseek.com/v1/messages", fallback_key))
            else:
                endpoints.append(("https://api.deepseek.com/v1/chat/completions", fallback_key))

        effective_timeout = timeout or self.timeout

        for url_idx, (target_url, active_key) in enumerate(endpoints):
            headers = self._build_headers(active_key or "")
            payload = self._build_payload(prompt, system_prompt, temperature, max_tokens=max_tokens)

            for attempt in range(1, max_retries + 1):
                try:
                    response = self.session.post(
                        target_url, headers=headers, json=payload, timeout=effective_timeout, verify=False
                    )
                    response.raise_for_status()

                    resp_data = response.json()

                    if resp_data.get("stop_reason") == "max_tokens":
                        logger.warning("LLM 输出被 max_tokens 截断，可能导致后续 JSON 或文本内容解析不全。")

                    text = self._parse_response(resp_data)
                    if not text:
                        logger.error(f"API 响应结构异常，文本内容为空: {resp_data}")
                        raise ValueError("LLM API 返回内容为空")

                    # 统计 token 用量
                    p_tokens, c_tokens = self._parse_usage(resp_data)
                    self.total_api_calls += 1
                    if p_tokens or c_tokens:
                        self.total_prompt_tokens += p_tokens
                        self.total_completion_tokens += c_tokens
                    else:
                        self.token_source = "estimated"

                        def estimate_tokens(t: str) -> int:
                            cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", t))
                            words = len(re.findall(r"\b[a-zA-Z0-9]+\b", t))
                            return int(cjk_chars * 1.5 + words * 1.3 + 1)

                        self.total_prompt_tokens += estimate_tokens(prompt) + estimate_tokens(system_prompt)
                        self.total_completion_tokens += estimate_tokens(text)

                    return text

                except Exception as e:
                    is_rate_limit = False
                    wait_time = 0

                    if "SSL" in str(e) or "Connection" in str(e) or "Timeout" in str(e):
                        logger.warning(f"检测到 SSL/TCP 连接异常 ({str(e)})，正在自动重建 Session 刷新连接...")
                        self._init_session()

                    if hasattr(e, "response") and e.response is not None:
                        status_code = getattr(e.response, "status_code", None)
                        if status_code in (401, 403):
                            if url_idx < len(endpoints) - 1:
                                logger.warning(
                                    f"端点 {target_url} 鉴权失败 (HTTP {status_code})，正在切换至备用端点..."
                                )
                                break
                            key_preview = active_key[:8] + "..." if active_key else "NOT_SET"
                            logger.error(
                                f"LLM API 鉴权失败 (HTTP {status_code})！Key: {key_preview}. "
                                f"格式: {self.api_format}. 请检查 .env LLM_API_KEY。"
                            )
                            raise RuntimeError(
                                f"LLM API 鉴权失败 (HTTP {status_code})。Key: {key_preview}。"
                                f"请检查 .env LLM_API_KEY 或设置 LLM_FALLBACK_API_KEY。"
                            ) from e
                        elif status_code == 429:
                            is_rate_limit = True
                            retry_after = e.response.headers.get("Retry-After")
                            if retry_after:
                                try:
                                    wait_time = int(retry_after)
                                except ValueError:
                                    pass

                    if is_rate_limit:
                        logger.warning(
                            f"触发 API 频率限制 (HTTP 429) (尝试 {attempt}/{max_retries})，正在执行退避重试..."
                        )
                    else:
                        logger.warning(f"LLM API 接入调用失败 ({target_url}) (尝试 {attempt}/{max_retries}): {str(e)}")

                    if attempt == max_retries:
                        if url_idx == len(endpoints) - 1:
                            raise RuntimeError(
                                f"All LLM API retry attempts failed ({max_retries} attempts). Last error: {e}"
                            ) from e
                        else:
                            logger.warning(f"端点 {target_url} 无法连接，正在自动切换至备用端点...")
                            break

                    sleep_time = wait_time if wait_time > 0 else (2**attempt) + random.uniform(0.5, 2.0)
                    if is_rate_limit and wait_time == 0:
                        sleep_time += 3.0
                    time.sleep(sleep_time)

        raise RuntimeError("All LLM API retry attempts failed")

    def extract_json_from_text(self, raw_output: str) -> Dict[str, Any]:
        """强力提取文本中的 JSON 部分，支持 markdown、object 和 array。"""
        # 剥离思维链输出（DeepSeek thinking 等），防止思考文本中的花括号干扰 JSON 边界定位
        raw_output = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()

        candidates = []

        code_block = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_output, re.DOTALL)
        if code_block:
            candidates.append(code_block.group(1))

        obj_start = raw_output.find("{")
        obj_end = raw_output.rfind("}")
        if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
            candidates.append(raw_output[obj_start : obj_end + 1])

        arr_start = raw_output.find("[")
        arr_end = raw_output.rfind("]")
        if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
            candidates.append(raw_output[arr_start : arr_end + 1])

        candidates.append(raw_output)

        for candidate in candidates:
            try:
                candidate_str = candidate.strip()
                if candidate_str:
                    return json.loads(candidate_str)
            except json.JSONDecodeError:
                continue

        # 记录原始输出片段便于诊断模型实际返回内容（如纯文本、截断、异常页面）
        preview = raw_output[:300].replace("\n", " ")
        logger.warning(f"JSON 提取失败，LLM 原始输出前300字符: {preview}")
        raise ValueError("LLM 返回内容中不包含任何合法的 JSON 结构")

    def call_json(
        self,
        prompt: str,
        system_prompt: str = "You are an AI research analyst. You must output valid JSON only. Do not wrap in markdown blocks.",
        temperature: float = 0.1,
        max_retries: int = 5,
    ) -> Dict[str, Any]:
        """请求并强力解析 JSON 输出，内置 JSON 损坏自动重试修复机制。"""
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
        """获取当前的 API 调用次数、Token 消耗及预估费用。"""
        pricing = self.MODEL_PRICING.get(
            self.model,
            {
                "input_per_1m_tokens": 0.14,
                "output_per_1m_tokens": 0.28,
            },
        )
        in_rate = pricing.get("input_per_1m_tokens", 0.14)
        out_rate = pricing.get("output_per_1m_tokens", 0.28)
        cost_usd = round(
            (self.total_prompt_tokens / 1_000_000.0) * in_rate
            + (self.total_completion_tokens / 1_000_000.0) * out_rate,
            6,
        )
        return {
            "total_api_calls": self.total_api_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "token_source": self.token_source,
            "estimated_cost_usd": cost_usd,
            "estimated_cost_cny": round(cost_usd * 7.2, 5) if cost_usd is not None else None,
        }

    def validate_connection(self) -> Dict[str, Any]:
        """启动时验证 API 连接与密钥有效性。返回诊断结果字典。"""
        result = {
            "api_key_configured": bool(self.api_key and self.api_key != "your_api_key_here"),
            "api_key_preview": (self.api_key[:8] + "...") if self.api_key else "NOT_SET",
            "base_url": self.base_url,
            "request_url": self.url,
            "api_format": self.api_format,
            "model": self.model,
            "timeout": self.timeout,
            "dns_patch_enabled": ENABLE_LLM_DNS_PATCH,
            "fallback_configured": bool(self.fallback_api_key),
            "connection_ok": False,
            "error": None,
        }
        if not result["api_key_configured"]:
            result["error"] = "LLM_API_KEY 未配置"
            return result
        try:
            self.call(
                prompt="Reply with exactly: OK",
                system_prompt="Connection test. Reply: OK",
                temperature=0.0,
                max_retries=1,
            )
            result["connection_ok"] = True
        except RuntimeError as e:
            result["error"] = str(e)
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {str(e)}"
        return result
