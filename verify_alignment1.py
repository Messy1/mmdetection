import torch
from transformers import AutoTokenizer

# 1. 初始化环境
model_path = '/ssd/wzh/models/LLM2CLIP-Llama-3.2-1B-Instruct-CC-Finetuned'
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

# 模拟 COCO 风格的拼接文本
# 注意：GroundingDINO 默认用 ". " 拼接类名
classes = ["papaya", "iron", "saxophone", "sneakers", "person"]
special_tokens = ". "
caption_string = special_tokens.join(classes) + "." 
# 结果应该是: "papaya. iron. saxophone. sneakers. person."

print(f"--- 待验证文本 ---\n'{caption_string}'\n")

# 模拟 tokens_positive 结构 (每个物体的字符区间)
# 手动计算一下区间 (实际源码中是自动算的)
tokens_positive = [
    [[0, 6]],    # papaya
    [[8, 12]],   # iron
    [[14, 23]],  # saxophone
    [[25, 33]],  # sneakers
    [[35, 41]]   # person
]

# 2. 模拟全文分词
tokenized_full = tokenizer([caption_string], return_tensors='pt')
input_ids = tokenized_full['input_ids'][0]
max_text_len = 256

# 3. 核心：更新后的 get_positive_map 逻辑
def simulate_new_get_positive_map(input_ids, tokens_positive, caption_string, tokenizer):
    positive_map = torch.zeros((len(tokens_positive), max_text_len), dtype=torch.float)
    ids_list = input_ids.tolist()
    
    for i, tok_pos in enumerate(tokens_positive):
        # 从字符区间提取原始单词
        char_start, char_end = tok_pos[0]
        target_word = caption_string[char_start:char_end]
        
        # 核心：尝试多种分词组合来对抗 Tiktoken 的空格合并
        # 模式 A: 独立单词 (可能不带空格)
        word_tokens_a = tokenizer(target_word, add_special_tokens=False).input_ids
        # 模式 B: 带前缀空格 (COCO 拼接常用模式)
        word_tokens_b = tokenizer(" " + target_word, add_special_tokens=False).input_ids
        
        found = False
        for search_ids in [word_tokens_b, word_tokens_a]:
            if found or len(search_ids) == 0: continue
            
            n = len(search_ids)
            # 在全文 ID 序列中滑动窗口搜索
            for k in range(len(ids_list) - n + 1):
                if ids_list[k : k + n] == search_ids:
                    positive_map[i, k : k + n] = 1.0
                    found = True
                    break
        
        if not found:
            print(f"⚠️ 警告: 目标 '{target_word}' 未能匹配到任何 Token！")
            
    return positive_map

# 4. 执行验证
pos_map = simulate_new_get_positive_map(input_ids, tokens_positive, caption_string, tokenizer)

# 5. 反向解码结果
print("--- 验证映射结果 ---")
for i, word_spec in enumerate(classes):
    active_indices = torch.nonzero(pos_map[i]).squeeze(-1)
    if len(active_indices) > 0:
        decoded = tokenizer.decode(input_ids[active_indices])
        print(f"物体 [{word_spec}] -> 匹配 Token 索引: {active_indices.tolist()} -> 解码结果: '{decoded}'")
    else:
        print(f"物体 [{word_spec}] -> ❌ 匹配失败")