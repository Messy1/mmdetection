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
    batch_size=16,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(type='ConcatDataset', datasets=[o365v1_od_dataset, flickr30k_dataset, gqa_dataset])
)

# ==================== 3. 优化器、策略与 Hook ====================
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper', 
    optimizer=dict(type='AdamW', lr=0.0004, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2), 
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            'text_feat_map': dict(lr_mult=10.0, decay_mult=1.0),
            'language_model': dict(lr_mult=0.0)  
        }
    )
)

max_epochs = 5
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[0, 3],
        gamma=0.1)
]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
auto_scale_lr = dict(enable=True, base_batch_size=64)

# 保留你原本优秀的 Checkpoint 存盘策略
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', 
        interval=500,    
        by_epoch=False,   
        max_keep_ckpts=3, 
        save_last=True    
    ),
    visualization=dict(type='GroundingVisualizationHook'))

load_from = '/ssd/wzh/models/groundingdino_swin-t_pruned_for_llm.pth'