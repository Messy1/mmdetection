_base_ = 'grounding_dino_swin-t_pretrain_obj365_goldg.py'

llm2clip_path = '/share/wzh/models/LLM2CLIP-Llama-3.2-1B-Instruct-CC-Finetuned'

# ==================== 1. 模型配置 ====================
model = dict(
    type='GroundingDINO',
    language_model=dict(
        _delete_=True,
        type='LLM2VecModel',
        model_name_or_path=llm2clip_path, 
        max_tokens=256,
        pad_to_max=False,
        use_peft=False  # LLM2CLIP 是全量模型
    ),
    text_mapper_hidden_dim=1024
)

# ==================== 2. 数据流与切词对齐 (最关键的一步) ====================
# 必须重写 train_pipeline，以确保数据预处理使用的是 LLM2CLIP 的 Tokenizer
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
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
    dict(
        type='RandomSamplingNegPos',
        tokenizer_name=llm2clip_path,  # 【核心修正】：统一使用 LLM2CLIP 分词器
        num_sample_negative=85,
        max_tokens=256),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities', 'tokens_positive', 'dataset_mode'))
]

# 将修正后的 pipeline 绑定到三个数据集上
o365v1_od_dataset = dict(
    type='ODVGDataset',
    data_root='data/objects365v1/',
    ann_file='o365v1_train_odvg.json',
    label_map_file='o365v1_label_map.json',
    data_prefix=dict(img='train/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline, 
    return_classes=True,
    backend_args=None)

flickr30k_dataset = dict(
    type='ODVGDataset',
    data_root='data/flickr30k_entities/',
    ann_file='final_flickr_separateGT_train_vg.json',
    label_map_file=None,
    data_prefix=dict(img='flickr30k_images/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline, 
    return_classes=True,
    backend_args=None)

gqa_dataset = dict(
    type='ODVGDataset',
    data_root='data/gqa/',
    ann_file='final_mixed_train_no_coco_vg.json',
    label_map_file=None,
    data_prefix=dict(img='images/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline, 
    return_classes=True,
    backend_args=None)

# 重新组装 DataLoader
train_dataloader = dict(
    _delete_=True,
    batch_size=8,  # FP32 安全起见用 4
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(type='ConcatDataset', datasets=[o365v1_od_dataset, flickr30k_dataset, gqa_dataset])
)

# ==================== 3. 优化器、策略与 Hook ====================
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper', 
    optimizer=dict(type='AdamW', lr=0.00004, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2), 
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),       
            'language_model': dict(lr_mult=0.0) ,
            'text_feat_map': dict(lr_mult=10.0) 
        }
    )
)

max_epochs = 4
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
auto_scale_lr = dict(enable=True, base_batch_size=64)

param_scheduler = [
    # 给重置了动量的优化器一个极其短暂的 500 步缓冲期，防止由于截断带来的初期 Loss 震荡
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        by_epoch=False,
        milestones=[5000,10000,12000], # 在第 1 个 Epoch 跑完时，再降维打击一次 (变为 0.000004)
        gamma=0.2)
]

# 保留你原本优秀的 Checkpoint 存盘策略
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', 
        interval=1000,    
        by_epoch=False,   
        max_keep_ckpts=3, 
        save_last=True
    ),
    visualization=dict(type='GroundingVisualizationHook'))

custom_hooks = [
    dict(
        type='CheckpointHook',
        interval=1,             # 每 1 个 Epoch 保存一次
        by_epoch=True,          # 按 Epoch 计算
        max_keep_ckpts=3,       # 最多保留 3 个 Epoch 权重
        out_dir='work_dirs/gdl2c2mlp/epoch_ckpts'
    )
]

load_from = '/share/wzh/models/groundingdino_swin-t_pruned_for_llm.pth'

env_cfg = dict(
    dist_cfg=dict(
        backend='nccl', 
        timeout=28800 # 8 Hours
    )
)