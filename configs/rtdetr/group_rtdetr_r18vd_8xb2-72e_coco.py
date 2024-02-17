_base_ = './rtdetr_r18vd_8xb2-72e_coco.py'

model = dict(type='GroupRTDETR', num_group=13)
