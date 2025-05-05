_base_ = './rtdetr_r18vd_8xb2-72e_coco.py'
pretrained = 'https://github.com/flytocc/mmdetection/releases/download/model_zoo/resnet18vd_pretrained_55f5a0d6.pth'  # noqa

model = dict(bbox_reparam=True)
