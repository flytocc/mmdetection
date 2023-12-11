_base_ = './rtdetr_r18vd_4xb4-72e_coco.py'

model = dict(
    type='DeepEquilibriumRTDETR',
    decoder=dict(
        num_layers=2,
        refinement_steps=10,
        supervision_position=[1, 3, 10],
        grad_accumulation=False,
        perturb_query_prob=0.2,
        perturb_query_intensity=0.1,
        perturb_ref_points_prob=0.2,
        perturb_ref_points_intensity=1 / 32,
        extra_supervisions_on_init_head=2,
        rag=2),
    bbox_head=dict(type='DeepEquilibriumRTDETRHead'))
