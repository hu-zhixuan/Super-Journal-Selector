import os
import json
import re
import time
import logging
import requests
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()


class LLMClient:
    """
    统一封装的 LLM 客户端，特别适配 Anthropic Messages API 格式 (/v1/messages)。
    支持强制直连（Bypass System Proxies）以解决 VPN 拦截问题，并提供强力 JSON 容错提取与指数重试。
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "https://fxb.supa.net.cn:6443")
        self.model = model or os.getenv("LLM_MODEL", "deepseek-v4-flash")

        # 规范化 Base URL 路径，确保以 /v1/messages 结尾
        self.url = self.base_url.rstrip("/")
        if not self.url.endswith("/v1/messages"):
            self.url = f"{self.url}/v1/messages"

        if not self.api_key or self.api_key == "your_api_key_here":
            logger.warning("未检测到有效的 LLM_API_KEY。若调用 API 将导致鉴权失败，请在 .env 中正确配置。")

    def call(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful academic research assistant.",
        temperature: float = 0.3,
        max_retries: int = 3,
    ) -> str:
        """
        发起 Anthropic Messages 协议请求，强制绕过本地系统代理
        """
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        payload = {
            "model": self.model,
            "max_tokens": 4000,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        # 显式声明直连 proxies，强行忽略系统环境变量中的代理 (如 HTTP_PROXY, HTTPS_PROXY)
        # 这对于使用全局代理但需要对国内 IP 直连的场景非常有效
        direct_proxies = {
            "http": None,
            "https": None
        }

        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(
                    self.url,
                    headers=headers,
                    json=payload,
                    proxies=direct_proxies,  # 强制直连
                    timeout=30
                )
                response.raise_for_status()
                
                resp_data = response.json()
                content_list = resp_data.get("content", [])
                if content_list and len(content_list) > 0:
                    text_content = content_list[0].get("text", "")
                    return text_content
                
                logger.error(f"API 响应结构异常: {resp_data}")
                raise ValueError("Anthropic API 返回内容为空")

            except Exception as e:
                logger.warning(f"LLM API 接入调用失败 (尝试 {attempt}/{max_retries}): {str(e)}")
                if attempt == max_retries:
                    raise
                time.sleep(2 ** attempt)
        return ""

    def call_json(
        self,
        prompt: str,
        system_prompt: str = "You are an AI research analyst. You must output valid JSON only. Do not wrap in markdown blocks.",
        temperature: float = 0.1,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """
        请求并强力解析 JSON 输出，内置对 Markdown 代码块的正则清洗容错。
        """
        raw_output = self.call(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_retries=max_retries,
        )

        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            pass

        json_match = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", raw_output, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        bracket_match = re.search(r"(\{.*\})", raw_output, re.DOTALL)
        if bracket_match:
            try:
                return json.loads(bracket_match.group(1))
            except json.JSONDecodeError:
                pass

        logger.error(f"无法从 LLM 返回内容中解析出有效 JSON:\n{raw_output[:500]}...")
        raise ValueError("LLM 返回结构不符合 JSON 格式约束")
