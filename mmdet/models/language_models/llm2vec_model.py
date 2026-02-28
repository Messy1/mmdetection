import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmdet.registry import MODELS
from transformers import AutoTokenizer

@MODELS.register_module()
class LLM2VecModel(BaseModule):
    def __init__(self,
                 model_name_or_path,
                 max_tokens=256,
                 pad_to_max=False,
                 use_peft=False, # 默认冻结，不训练LLM
                 **kwargs):
        super().__init__()
        self.max_tokens = max_tokens
        self.pad_to_max = pad_to_max
        
        # 1. 加载 Fast Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        # LLM (如LLaMA) 默认没有 pad_token，必须设置，否则无法 batch 处理
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2. 加载 LLM2Vec 模型
        from llm2vec import LLM2Vec
        self.model = LLM2Vec.from_pretrained(
            model_name_or_path,
            enable_bidirectional=True,
            peft_model_name_or_path=kwargs.get('peft_path', None)
        )
        
        # 3. 冻结模型，转为半精度以节省显存
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 可选：如果显卡支持，可以转为 bfloat16
        # self.model.to(torch.bfloat16)

    @property
    def language_dim(self):
        # 返回 LLM 的隐藏层维度 (比如 LLaMA-3-8B 是 4096)
        return self.model.config.hidden_size

    def forward(self, texts, **kwargs):
        """
        texts: list[str], 批量文本
        """
        device = next(self.model.parameters()).device
        
        # Tokenization
        inputs = self.tokenizer(
            texts,
            padding="max_length" if self.pad_to_max else True,
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
            return_offsets_mapping=True # 如果下游需要用到 offset 这里可以返回
        ).to(device)

        # 获取特征
        with torch.no_grad(): # 确保不计算梯度
            outputs = self.model.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True
            )
        
        # 取最后一层 hidden states
        hidden_states = outputs.hidden_states[-1].float()

        mask = inputs['attention_mask'].bool()

        if mask.dim() == 2:
            embedded = hidden_states * mask.unsqueeze(-1).float()
        else:
            embedded = hidden_states

        bs, seq_len = mask.shape
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(bs, -1)
        text_self_attn_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        eye_mask = torch.eye(seq_len, dtype=torch.bool, device=device).unsqueeze(0)
        text_self_attn_mask = text_self_attn_mask | eye_mask

        results = {
            'embedded': embedded,          # DINO 用于跨模态融合的核心特征 (Padding处为0)
            'masks': text_self_attn_mask,                 # 注意力掩码 (原版在 sub_sentence 时是 3D，我们退回 2D 对自然句子更鲁棒)
            'hidden': hidden_states,       # 原始的 hidden states
            'position_ids': position_ids,  # 位置编码 ID
            'text_token_mask': mask        # 区分有效词和 Pad 的掩码
        }
        
        # 如果下游还需要 offset_mapping，我们可以通过一个特殊的私有变量塞出去
        # 因为原版的 tokenized 对象是在 get_tokens_and_prompts 里管理的
        
        return results