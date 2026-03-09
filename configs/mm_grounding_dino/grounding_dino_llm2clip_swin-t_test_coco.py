# 1. 直接继承你刚刚微调用的配置文件（这样模型结构、LLM路径会自动对齐）
_base_ = './grounding_dino_llm2clip_swin-t_pretrain.py'

# ==================== COCO 数据集与评测配置 ====================
dataset_type = 'CocoDataset'
data_root = 'data/coco/'  # 请确保你的 COCO 数据集在这个路径下

test_pipeline = [
    dict(
        type='LoadImageFromFile', backend_args=None,
        imdecode_backend='pillow'),
    dict(
        type='FixScaleResize',
        scale=(800, 1333),
        keep_ratio=True,
        backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities',
                   'tokens_positive'))
]

# 重新定义验证/测试的 DataLoader
val_dataloader = dict(
    batch_size=2,  # LLM 推理时显存占用大，测试时保持 bs=1 最安全
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        return_classes=True  # 【关键】：必须为 True，自动生成 COCO 80类的文本 Prompt
    )
)

test_dataloader = val_dataloader

# 重新定义评估器 (Evaluator) 为 COCO 标准的 mAP 计算器
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox',
    format_only=False)

test_evaluator = val_evaluator