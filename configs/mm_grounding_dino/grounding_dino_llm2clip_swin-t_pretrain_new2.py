_base_ = 'grounding_dino_swin-t_pretrain_obj365_goldg.py'

llm2clip_path = '/ssd/wzh/models/LLM2CLIP-Llama-3.2-1B-Instruct-CC-Finetuned'

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
    )
)

# ==================== 2. 数据流与切词对齐 ====================
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
        tokenizer_name=llm2clip_path,  # 统一使用 LLM2CLIP 分词器
        num_sample_negative=85,
        max_tokens=256),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities', 'tokens_positive', 'dataset_mode'))
]

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

train_dataloader = dict(
    _delete_=True,
    batch_size=16,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(type='ConcatDataset', datasets=[o365v1_od_dataset, flickr30k_dataset, gqa_dataset])
)

# ==================== 3. Stage 2 微调专用优化器与策略 ====================
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper', 
    # 🔥 核心改变 1：基础学习率直降 10 倍 (0.0004 -> 0.00004)
    optimizer=dict(type='AdamW', lr=0.00004, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2), 
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            # 🔥 核心改变 2：投影层继续保持 10 倍优势，吃透这波微调红利
            'text_feat_map': dict(lr_mult=10.0, decay_mult=1.0),
            'language_model': dict(lr_mult=0.0)  
        }
    )
)

# 🔥 核心改变 3：重塑训练生命周期 (再跑 2 个 Epoch 压榨极限)
max_epochs = 2 
param_scheduler = [
    # 给重置了动量的优化器一个极其短暂的 500 步缓冲期，防止由于截断带来的初期 Loss 震荡
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[1], # 在第 1 个 Epoch 跑完时，再降维打击一次 (变为 0.000004)
        gamma=0.1)
]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
auto_scale_lr = dict(enable=True, base_batch_size=64)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', 
        interval=500,    
        by_epoch=False,   
        max_keep_ckpts=3, 
        save_last=True    
    ),
    visualization=dict(type='GroundingVisualizationHook'))

# 🔥 核心改变 4：精准对接刚刚跑完的 10500 步断点
load_from = '/home/wanzhiheng/workspace/mmdetection/work_dirs/grounding_dino_llm2clip_swin-t_pretrain_new/iter_10500.pth'