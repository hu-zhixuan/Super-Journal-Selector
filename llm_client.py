import os
import json
import re
import time
import logging
import requests
import urllib3
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# 禁用 urllib3 的证书警告 (因为我们使用 IP 直连时会关闭 verify 校验)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()


class LLMClient:
    """
    统一封装的 LLM 客户端，适配 Anthropic Messages API 格式 (/v1/messages)。
    特制版：采用物理 IP 直连 + Host 头部绑定方案，彻底击穿 NekoRay/Clash 的 Fake-IP 路由劫持。
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "https://fxb.supa.net.cn:6443")
        self.model = model or os.getenv("LLM_MODEL", "deepseek-v4-flash")

        # 物理 IP 直连转换：如果域名是 fxb.supa.net.cn，强行重定向到其真实的物理 IP: 114.80.15.146
        # 从而避开 198.18.x.x Fake-IP 劫持，让全局代理网卡的“绕过大陆”规则 100% 触发直连
        self.use_ip_direct = "fxb.supa.net.cn" in self.base_url
        if self.use_ip_direct:
            logger.info("检测到目标域名 fxb.supa.net.cn，启动物理 IP 直连与 Host 绑定方案以绕过 Fake-IP 劫持。")
            self.url = "https://114.80.15.146:6443/v1/messages"
        else:
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
        发起 Anthropic Messages 协议请求，强制绕过本地系统代理并支持物理 IP 直连
        """
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        # 如果启用物理 IP 直连，必须手动绑定 Host 头部，以便 CDN/反向代理正确分发路由
        if self.use_ip_direct:
            headers["Host"] = "fxb.supa.net.cn"

        payload = {
            "model": self.model,
            "max_tokens": 4000,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        # 显式声明忽略本地 HTTP 代理变量
        direct_proxies = {
            "http": None,
            "https": None
        }

        for attempt in range(1, max_retries + 1):
            try:
                # verify=False 保证使用 IP 直连时不会因为 SSL 证书域名不匹配而报错
                response = requests.post(
                    self.url,
                    headers=headers,
                    json=payload,
                    proxies=direct_proxies,
                    verify=False,  # 物理 IP 直连防证书报错
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
