import os
# 阻断死锁与联网
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import cv2
cv2.setNumThreads(0) # 防止 OpenCV 多线程死锁

import torch
from mmdet.apis import init_detector
from mmengine.dataset import Compose
from mmengine.dataset.utils import pseudo_collate
from mmdet.registry import VISUALIZERS

# ================= 1. 配置路径 =================
config_file = 'configs/mm_grounding_dino/grounding_dino_llm2clip_swin-t_test_coco.py'
checkpoint_file = 'work_dirs/grounding_dino_llm2clip_swin-t_pretrain/iter_21500.pth'
img_path = '/ssd/wzh/workspace/mmdetection/data/coco/val2017/000000000139.jpg' 
text_prompt = 'A person.'

print("=== 1. 加载模型 ===")
model = init_detector(config_file, checkpoint_file, device='cuda:0')
model.eval() # 确保进入测试模式

print("=== 2. 手动执行数据预处理 (彻底绕过 inference_detector 的死锁坑) ===")
# 过滤掉 LoadAnnotations，因为单图推理没有标注文件
clean_pipeline = []
for step in model.cfg.test_pipeline:
    if step['type'] != 'LoadAnnotations':
        clean_pipeline.append(step)
        
pipeline = Compose(clean_pipeline)
data = dict(img_path=img_path, img_id=0, text=text_prompt, custom_entities=True)
data = pipeline(data)

print("=== 3. 移至 GPU 并推理 (FP16 防护罩) ===")
data = pseudo_collate([data])
data['inputs'] = [inp.to('cuda:0') for inp in data['inputs']]
data['data_samples'] = [sample.to('cuda:0') for sample in data['data_samples']]

# 【关键修复】：将 dtype 改为 torch.float16，这是所有 CUDA 算子都认识的半精度！
with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
    result = model.test_step(data)[0] 

# 安全兜底，防止极个别噪点引发的得分 NaN
result.pred_instances.scores = torch.nan_to_num(result.pred_instances.scores, nan=0.0)

# 【关键修复】：将 dtype 改为 torch.float16，这是所有 CUDA 算子都认识的半精度！
with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
    result = model.test_step(data)[0] 

# =====================================================================
# 【救命神药】：脱离模型后，立刻将 FP16 强制转回 FP32，彻底斩断面积计算溢出！
# =====================================================================
result.pred_instances.bboxes = result.pred_instances.bboxes.to(torch.float32)
result.pred_instances.scores = result.pred_instances.scores.to(torch.float32)

# 安全兜底，防止极个别噪点引发的得分 NaN
result.pred_instances.scores = torch.nan_to_num(result.pred_instances.scores, nan=0.0)

print("=== 4. 正在清洗异常坐标并准备画图 ===")
bboxes = result.pred_instances.bboxes
scores = result.pred_instances.scores

# 1. 找出 bboxes 和 scores 中没有任何 NaN/Inf 的合法索引
valid_bbox_mask = ~torch.isnan(bboxes).any(dim=-1) & ~torch.isinf(bboxes).any(dim=-1)
valid_score_mask = ~torch.isnan(scores)

# 2. 【新增防线】：过滤掉宽或高 <= 0 的“倒挂框”（这正是导致字体开方出 NaN 的元凶！）
widths = bboxes[:, 2] - bboxes[:, 0]
heights = bboxes[:, 3] - bboxes[:, 1]
valid_area_mask = (widths > 0) & (heights > 0)

# 3. 取交集，只保留完全健康的预测实例
valid_mask = valid_bbox_mask & valid_score_mask & valid_area_mask
result.pred_instances = result.pred_instances[valid_mask]

# 更新清洗后的 scores 和 bboxes 变量，用于后续打印
scores = result.pred_instances.scores
bboxes = result.pred_instances.bboxes
# =====================================================================

# 看看模型给出的最高分到底是多少！
if len(scores) > 0:
    max_score = scores.max().item()
    print(f"\n模型给出的全局最高置信度: {max_score:.4f}")

    # 强行取出得分排名前 5 的框
    topk_num = min(5, len(scores))
    topk_scores, topk_idx = torch.topk(scores, topk_num)

    print("\n--- 得分 Top-5 的预测结果 ---")
    for i in range(topk_num):
        print(f"第 {i+1} 名 | 得分: {topk_scores[i].item():.4f} | 框坐标: {bboxes[topk_idx[i]].int().tolist()}")
else:
    max_score = 0.0
    print("\n⚠️ 警告：所有框都被过滤掉了，没有有效的预测结果。")

# 准备画图
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.dataset_meta = model.dataset_meta

out_filename = 'result_demo.jpg'

# 动态阈值：取 0.005 和 (最高分 - 0.002) 中的较小值，确保一定能画出最高分的框
vis_thr = min(0.005, max(0.0, max_score - 0.002)) if max_score > 0 else 0.001
print(f"\n正在以 {vis_thr:.4f} 的动态超低阈值生成可视化图片...")

visualizer.add_datasample(
    name='result',
    image=img,
    data_sample=result,
    draw_gt=False,
    show=False, 
    pred_score_thr=vis_thr,  # 使用安全的低阈值
    out_file=out_filename
)

print(f"🎉 大功告成！新图片已覆盖保存至：{out_filename}")