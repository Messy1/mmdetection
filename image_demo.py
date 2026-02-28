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

config_file = 'configs/mm_grounding_dino/grounding_dino_llm2clip_swin-t_test_coco.py'
checkpoint_file = 'work_dirs/grounding_dino_llm2clip_swin-t_pretrain/iter_20000.pth'
img_path = '/ssd/wzh/workspace/mmdetection/data/coco/val2017/000000000139.jpg' 
text_prompt = 'A person and a television in the room.'

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

# 安全兜底，防止极个别噪点引发的 NaN
result.pred_instances.scores = torch.nan_to_num(result.pred_instances.scores, nan=0.0)
print("=== 4. 开始可视化画图 ===")
# ... 往下保留你原来的画图代码 ...
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.dataset_meta = model.dataset_meta

out_filename = 'result_demo.jpg'
visualizer.add_datasample(
    name='result',
    image=img,
    data_sample=result,
    draw_gt=False,
    show=False, 
    pred_score_thr=0.3, # 只画出得分超过 0.3 的框
    out_file=out_filename
)

print(f"🎉 大功告成！图片已保存至：{out_filename}")

# 打印一下找到的高置信度框
# 获取所有框的得分
scores = result.pred_instances.scores
bboxes = result.pred_instances.bboxes

# 看看模型给出的最高分到底是多少！
print(f"\n模型给出的全局最高置信度: {scores.max().item():.4f}")

# 强行取出得分排名前 5 的框（不管它有没有超过 0.3）
topk_scores, topk_idx = torch.topk(scores, min(5, len(scores)))

print("\n--- 得分 Top-5 的预测结果 ---")
for i in range(len(topk_scores)):
    print(f"第 {i+1} 名 | 得分: {topk_scores[i].item():.4f} | 框坐标: {bboxes[topk_idx[i]].int().tolist()}")

# 为了能在图上看到结果，把可视化阈值强行降到比最高分低一点点
# 例如，如果最高分是 0.15，我们就把阈值设为 0.1
vis_thr = max(0.05, scores.max().item() - 0.05) 
print(f"\n正在以 {vis_thr:.4f} 的阈值重新生成可视化图片...")

visualizer.add_datasample(
    name='result',
    image=img,
    data_sample=result,
    draw_gt=False,
    show=False, 
    pred_score_thr=vis_thr,  # <--- 使用动态超低阈值
    out_file=out_filename
)
print(f"🎉 新图片已覆盖保存至：{out_filename}")