# ==============================================================================
# 继承原版大配置 
# ==============================================================================
_base_ = './grounding_dino_llm2clip_swin-t_test_coco.py' 

data_root = 'data/coco/'

# ==============================================================================
# 1. 纯净版数据增强 Pipeline (剔除导致崩溃的文本算子，保留视觉多尺度)
# ==============================================================================
mini_train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None, imdecode_backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    # 🔥 这里删除了原版的文本负样本采样算子，彻底规避 Tuple 报错
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities')) # 🔥 移除了 'tokens_positive'
]

mini_test_pipeline = [
    dict(
        type='LoadImageFromFile', 
        backend_args=None,
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
                   'scale_factor', 'text', 'custom_entities')) # 🔥 移除了 'tokens_positive'
]

# ==============================================================================
# 2. 数据集重定向与加载器配置 (抛弃 ConcatDataset，使用纯净 CocoDataset)
# ==============================================================================
train_dataloader = dict(
    _delete_=True,
    batch_size=4,  # 根据你的显存调整
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='CocoDataset', # 🔥 强制使用 CocoDataset
        data_root=data_root,
        ann_file='annotations/mini_instances_train2017.json', 
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=mini_train_pipeline, # 🔥 注入自定义的纯净训练管道
        return_classes=True
    )
)

val_dataloader = dict(
    _delete_=True,
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset', # 🔥 强制使用 CocoDataset
        data_root=data_root,
        ann_file='annotations/mini_instances_val2017.json', 
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=mini_test_pipeline, # 🔥 注入自定义的纯净测试管道
        return_classes=True
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='CocoMetric',
    ann_file=data_root + 'annotations/mini_instances_val2017.json', 
    metric='bbox',
    format_only=False
)
test_evaluator = val_evaluator

# ==============================================================================
# 3. 核心排雷：非对称学习率设置 (解决 "text_feat_map 随机层停滞" 问题)
# ==============================================================================
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001), 
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            # 完全冻结语言模型，保护大模型脆弱的高维特征空间
            'language_model': dict(lr_mult=0.0, decay_mult=0.0),
            
            # 视觉底座保持极小步伐微调 (跟随自动缩放后的 lr)
            'backbone': dict(lr_mult=0.0, decay_mult=0.0),
            
            # 🔥 【破局核心】：给随机初始化的投影层 10 倍学习率！
            'text_feat_map': dict(lr_mult=10.0, decay_mult=1.0),
            
            # 绝对位置编码无需正则化
            'absolute_pos_embed': dict(decay_mult=0.0),
            'backbone.patch_embed.norm': dict(decay_mult=0.0),
            'backbone.norm': dict(decay_mult=0.0),
        }
    )
)

# 开启自动学习率缩放（基础 BS 为 64，严格匹配上面的 lr=1e-4）
auto_scale_lr = dict(enable=True, base_batch_size=64)

# ==============================================================================
# 4. 训练节奏调整 (适配小数据集)
# ==============================================================================
max_epochs = 12 

train_cfg = dict(
    type='EpochBasedTrainLoop', 
    max_epochs=max_epochs, 
    val_interval=1 
)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

default_hooks = dict(logger=dict(type='LoggerHook', interval=10))