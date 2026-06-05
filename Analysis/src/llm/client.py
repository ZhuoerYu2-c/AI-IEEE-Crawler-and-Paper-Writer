"""
MiniMax LLM 客户端（OpenAI 兼容格式，支持流式输出）
"""
import time
import requests
from typing import List, Dict, Iterator


class MiniMaxClient:
    def __init__(self, api_key: str, base_url: str, model: str, timeout_s: int = 240):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 3000,
        temperature: float = 0.2,
        max_retries: int = 3,
        retry_wait_s: int = 8,
    ) -> str:
        """调用 MiniMax LLM 生成文本（非流式）"""
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_err = None

        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_s,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]

            except requests.exceptions.ReadTimeout as e:
                last_err = e
                print(f"\n⚠️ 第 {attempt}/{max_retries} 次请求超时...")
            except requests.exceptions.ConnectionError as e:
                last_err = e
                print(f"\n⚠️ 第 {attempt}/{max_retries} 次连接失败...")
            except requests.exceptions.HTTPError as e:
                last_err = e
                print(f"\n⚠️ 第 {attempt}/{max_retries} 次 HTTP 错误：{e}")
                try:
                    print(f"响应内容: {resp.text[:500]}")
                except:
                    pass
            except Exception as e:
                last_err = e
                print(f"\n⚠️ 第 {attempt}/{max_retries} 次错误：{e}")

            if attempt < max_retries:
                print(f"⏳ {retry_wait_s} 秒后重试...")
                time.sleep(retry_wait_s)

        raise RuntimeError(f"LLM 请求失败，已重试 {max_retries} 次。最后错误：{last_err}")

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 3000,
        temperature: float = 0.2,
        max_retries: int = 3,
        retry_wait_s: int = 8,
    ) -> Iterator[str]:
        """
        调用 MiniMax LLM 生成文本（流式输出）
        Yields: 逐字/逐句返回生成内容
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,  # 启用流式
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_err = None

        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_s,
                    stream=True,  # 启用流式响应
                )
                resp.raise_for_status()

                # 流式处理 SSE 格式
                buffer = ""
                for line in resp.iter_lines(decode_unicode=True):
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            import json as json_lib
                            data = json_lib.loads(data_str)
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                buffer += content
                                # 每积累一定字符或遇到换行就 yield
                                if len(buffer) >= 10 or content.endswith("\n"):
                                    yield buffer
                                    buffer = ""
                        except:
                            continue
                
                # 返回剩余 buffer
                if buffer:
                    yield buffer
                return

            except requests.exceptions.ReadTimeout as e:
                last_err = e
                print(f"\n⚠️ 第 {attempt}/{max_retries} 次请求超时...")
            except requests.exceptions.ConnectionError as e:
                last_err = e
                print(f"\n⚠️ 第 {attempt}/{max_retries} 次连接失败...")
            except Exception as e:
                last_err = e
                print(f"\n⚠️ 第 {attempt}/{max_retries} 次错误：{e}")

            if attempt < max_retries:
                print(f"⏳ {retry_wait_s} 秒后重试...")
                time.sleep(retry_wait_s)

        raise RuntimeError(f"LLM 流式请求失败，已重试 {max_retries} 次。最后错误：{last_err}")


def stream_chat(
    client: MiniMaxClient,
    messages: List[Dict[str, str]],
    max_tokens: int = 3000,
    temperature: float = 0.2,
) -> str:
    """
    执行流式聊天并收集所有输出
    实时打印到终端，便于监控
    """
    print("\n" + "=" * 60)
    print("📤 AI 正在生成...")
    print("=" * 60)
    
    collected = []
    for chunk in client.chat_stream(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    ):
        print(chunk, end="", flush=True)
        collected.append(chunk)
    
    print("\n" + "=" * 60)
    print("✅ 生成完成")
    print("=" * 60)
    
    return "".join(collected)
