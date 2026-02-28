import torch
from transformers import AutoTokenizer

# 1. 加载你的基础 LLaMA 分词器
model_path = '/ssd/wzh/models/Sheared-LLaMA-1.3B' # 替换为你的 Base 模型路径
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. 假设输入了一段文本和对应的字符级标注 (GT)
text = "papaya. iron. saxophone. sneakers. person."
print(f"原始文本: '{text}'")

# 我们假设两个物体的字符区间（根据之前你的输出）
# sneakers: 索引 25 到 33
# person:   索引 35 到 41
tokens_positive = [
    [[25, 33]],  # Object 1 (sneakers)
    [[35, 41]]   # Object 2 (person)
]

# 3. 模拟 Grounding DINO 里的处理逻辑
tokenized = tokenizer([text], return_tensors='pt', return_offsets_mapping=True)
offset_mapping = tokenized['offset_mapping'][0]

# 模拟我们写的 overlap > 0 的 get_positive_map 逻辑
max_text_len = 256
positive_map = torch.zeros((len(tokens_positive), max_text_len), dtype=torch.float)

for j, tok_pos in enumerate(tokens_positive):
    for (char_start, char_end) in tok_pos:
        for token_idx, offset in enumerate(offset_mapping):
            token_start, token_end = offset[0].item(), offset[1].item()
            if token_start == token_end: # 过滤特殊字符
                continue
            
            # 计算交集
            overlap = max(0, min(token_end, char_end) - max(token_start, char_start))
            if overlap > 0:
                positive_map[j, token_idx] = 1.0

# 4. 见证奇迹：反向解码激活的 Token
input_ids = tokenized['input_ids'][0]

for i in range(len(tokens_positive)):
    # 找到 positive_map 中值为 1 的 token 索引
    active_token_indices = torch.nonzero(positive_map[i]).squeeze(-1)
    
    # 提取这些索引对应的 input_ids
    active_token_ids = input_ids[active_token_indices]
    
    # 用 tokenizer 解码回字符串
    decoded_text = tokenizer.decode(active_token_ids)
    print(f"-> 目标 {i+1} 字符区间 {tokens_positive[i]} 映射到的 Token 解码结果: '{decoded_text}'")