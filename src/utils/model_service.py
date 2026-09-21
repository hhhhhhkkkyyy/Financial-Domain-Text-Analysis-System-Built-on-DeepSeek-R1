"""
模型服务部署模块 - 基于Transformers部署DeepSeek-R1-Distill-Qwen-1.5B
Model Service Deployment Module (Transformers only, no vLLM)
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Generator, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """模型配置"""
    name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    local_path: str = ""  # 用户自定义模型路径
    device_map: str = "auto"
    torch_dtype: str = "float16"
    temperature: float = 0.3
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    max_new_tokens: int = 2048
    do_sample: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False


class ModelService:
    """
    模型服务类 - 基于Transformers直接推理
    不使用vLLM，纯Transformers本地推理
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.model = None
        self.tokenizer = None
        self._initialized = False

    def initialize(self) -> bool:
        """使用Transformers加载模型（优先用户本地路径）"""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_path = ""
            if self.config.local_path and os.path.exists(self.config.local_path):
                model_path = self.config.local_path
                logger.info(f"从用户指定路径加载模型: {model_path}")
            elif os.path.exists(self.config.name):
                model_path = self.config.name
                logger.info(f"从本地路径加载模型: {model_path}")
            else:
                model_path = self.config.name
                logger.info(f"从HuggingFace加载模型: {model_path}")

            logger.info(f"正在加载模型: {model_path}")

            if self.config.torch_dtype == "float16":
                torch_dtype = torch.float16
            elif self.config.torch_dtype == "bfloat16":
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )

            load_kwargs = {
                "device_map": self.config.device_map,
                "torch_dtype": torch_dtype,
                "trust_remote_code": True,
            }

            if self.config.load_in_8bit:
                load_kwargs["load_in_8bit"] = True
            elif self.config.load_in_4bit:
                load_kwargs["load_in_4bit"] = True
                load_kwargs["bnb_4bit_compute_dtype"] = torch.float16

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **load_kwargs
            )

            self._initialized = True
            logger.info("模型加载成功 (Transformers模式)")
            return True

        except Exception as e:
            logger.error(f"Transformers模型加载失败: {e}")
            return False

    def generate(self, prompt: str, **kwargs) -> str:
        """统一生成接口"""
        if not self._initialized:
            raise RuntimeError("模型未初始化")
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.config.max_new_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "repetition_penalty": kwargs.get("repetition_penalty", self.config.repetition_penalty),
            "do_sample": kwargs.get("do_sample", self.config.do_sample),
        }

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        return response

    def batch_generate(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """批量生成，记录每次调用的耗时和状态"""
        results = []
        for i, prompt in enumerate(prompts):
            start_time = time.time()
            try:
                response = self.generate(prompt, **kwargs)
                elapsed = time.time() - start_time
                results.append({
                    'index': i,
                    'prompt': prompt[:200] + '...' if len(prompt) > 200 else prompt,
                    'response': response,
                    'status': 'success',
                    'elapsed_seconds': elapsed,
                    'response_length': len(response)
                })
                logger.info(f"批量推理 [{i+1}/{len(prompts)}] 完成, 耗时: {elapsed:.2f}s")
            except Exception as e:
                elapsed = time.time() - start_time
                results.append({
                    'index': i,
                    'status': 'error',
                    'error': str(e),
                    'elapsed_seconds': elapsed
                })
                logger.error(f"批量推理 [{i+1}/{len(prompts)}] 失败: {e}")
        return results

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        info = {
            'model_name': self.config.name,
            'initialized': self._initialized,
            'deployment_mode': 'Transformers',
        }
        if self.model:
            try:
                info['device'] = str(self.model.device)
                info['dtype'] = str(self.model.dtype)
            except:
                pass
        return info

    def cleanup(self):
        """清理模型资源"""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        self._initialized = False
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        logger.info("模型资源已清理")
