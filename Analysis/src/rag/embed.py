"""
Qwen Embedding API 封装（带进度条）
"""
from __future__ import annotations

import requests
from typing import List
from tqdm import tqdm


class QwenEmbedder:
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-v4",
        dimensions: int = 1024,
        batch_size: int = 10,
        show_progress: bool = True,
    ):
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        将文本列表转换为向量
        """
        embeddings = []
        
        # 分批处理
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        iterator = range(0, len(texts), self.batch_size)
        if self.show_progress:
            iterator = tqdm(
                iterator,
                desc="🔢 生成向量",
                unit="批",
                total=total_batches,
            )
        
        for i in iterator:
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._encode_batch(batch)
            embeddings.extend(batch_embeddings)
        
        return embeddings

    def _encode_batch(self, texts: List[str]) -> List[List[float]]:
        """单批编码"""
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model,
            "input": texts,
            "dimensions": self.dimensions,
            "encoding_format": "float",
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        
        # 按索引排序确保顺序一致
        sorted_embeddings = sorted(
            data["data"], 
            key=lambda x: x["index"]
        )
        
        return [item["embedding"] for item in sorted_embeddings]

    def encode_single(self, text: str) -> List[float]:
        """单个文本编码"""
        result = self.encode([text])
        return result[0] if result else []
