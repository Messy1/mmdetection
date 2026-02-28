import torch

# 1. 填入你手里现有的完整 mmgroundingdino-t 权重路径
official_ckpt_path = '/ssd/wzh/models/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth'
old_ckpt = torch.load(official_ckpt_path, map_location='cpu')

new_state_dict = {}

# 如果你的权重里最外层没有 'state_dict' 这个键，可以直接遍历 old_ckpt
# MMDetection 官方的权重通常都包裹在 'state_dict' 里面
state_dict = old_ckpt.get('state_dict', old_ckpt)

for k, v in state_dict.items():
    # 丢弃旧的 BERT 语言模型权重
    if k.startswith('language_model.'):
        continue
    # 丢弃旧的 768->256 线性投影层权重
    if k.startswith('text_feat_map.'):
        continue
    
    # 其它所有视觉主干、跨模态编码器、检测头的权重全盘保留！
    new_state_dict[k] = v

# 替换回原来的字典结构
if 'state_dict' in old_ckpt:
    old_ckpt['state_dict'] = new_state_dict
else:
    old_ckpt = new_state_dict

# 2. 保存为你专用的“纯净版”初始权重
new_ckpt_path = '/ssd/wzh/models/groundingdino_swin-t_pruned_for_llm.pth'
torch.save(old_ckpt, new_ckpt_path)
print(f"清洗完成！新的权重已保存至: {new_ckpt_path}")