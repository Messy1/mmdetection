import json
import random
import os

def create_mini_coco(src_json, dst_json, sample_ratio=0.05, seed=42):
    print(f"正在读取 {src_json}...")
    with open(src_json, 'r') as f:
        coco = json.load(f)

    # 随机抽样
    random.seed(seed)
    num_samples = int(len(coco['images']) * sample_ratio)
    sampled_images = random.sample(coco['images'], num_samples)
    
    # 获取抽样图片的 ID 集合，用于过滤标注
    sampled_image_ids = set(img['id'] for img in sampled_images)
    sampled_annotations = [ann for ann in coco['annotations'] if ann['image_id'] in sampled_image_ids]

    # 构建全新的 JSON 结构
    mini_coco = {
        'info': coco.get('info', {}),
        'licenses': coco.get('licenses', []),
        'categories': coco['categories'],
        'images': sampled_images,
        'annotations': sampled_annotations
    }

    print(f"抽样完成: 提取了 {len(sampled_images)} 张图片 和 {len(sampled_annotations)} 个目标框。")
    with open(dst_json, 'w') as f:
        json.dump(mini_coco, f)
    print(f"✅ 子集已保存至: {dst_json}\n")

if __name__ == '__main__':
    # 请确认这里的 data_root 是你真实的 COCO 标注路径
    data_root = '/ssd/wzh/workspace/mmdetection/data/coco/annotations/'
    
    # 1. 抽样训练集 (11万张图的 5% 大约是 5900 张图)
    create_mini_coco(
        src_json=os.path.join(data_root, 'instances_train2017.json'), 
        dst_json=os.path.join(data_root, 'mini_instances_train2017.json'), 
        sample_ratio=0.05
    )
    
    # 2. 抽样验证集 (5000张图的 20% 大约是 1000 张图)
    create_mini_coco(
        src_json=os.path.join(data_root, 'instances_val2017.json'), 
        dst_json=os.path.join(data_root, 'mini_instances_val2017.json'), 
        sample_ratio=0.20
    )