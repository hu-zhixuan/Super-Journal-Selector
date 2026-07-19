import os
import json
import re
import time
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()


class LLMClient:
    """
    统一封装的 LLM 调用客户端，支持 OpenAI 兼容 API（DeepSeek, Qwen, GLM, local Ollama 等）。
    提供结构化 JSON 提取能力及指数退避重试机制。
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")

        if not self.api_key or self.api_key == "your_api_key_here":
            logger.warning("未检测到有效的 LLM_API_KEY。若调用 API 将导致鉴权失败，请在 .env 中正确配置。")

        self.client = OpenAI(api_key=self.api_key or "mock-key", base_url=self.base_url)

    def call(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful academic research assistant.",
        temperature: float = 0.3,
        max_retries: int = 3,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        发起普通文本生成或 JSON 请求
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        for attempt in range(1, max_retries + 1):
            try:
                kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                }
                if response_format:
                    # 部分兼容 API (如 DeepSeek-V3/GPT) 支持 json_object
                    kwargs["response_format"] = response_format

                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                return content if content else ""
            except OpenAIError as e:
                logger.warning(f"LLM API 调用失败 (尝试 {attempt}/{max_retries}): {str(e)}")
                if attempt == max_retries:
                    raise
                time.sleep(2 ** attempt)
        return ""

    def call_json(
        self,
        prompt: str,
        system_prompt: str = "You are an AI research analyst. You must output valid JSON only without markdown code blocks.",
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
            response_format={"type": "json_object"}
        )

        # 尝试直接解析 JSON
        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            pass

        # 若失败，尝试清洗 Markdown 包装 ```json ... ``` 或 ``` ... ```
        json_match = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", raw_output, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # 若仍失败，尝试提取首个 { 到最后一个 }
        bracket_match = re.search(r"(\{.*\})", raw_output, re.DOTALL)
        if bracket_match:
            try:
                return json.loads(bracket_match.group(1))
            except json.JSONDecodeError:
                pass

        logger.error(f"无法从 LLM 返回内容中解析出有效 JSON:\n{raw_output[:500]}...")
        raise ValueError("LLM 返回结构不符合 JSON 格式约束")
