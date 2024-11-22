import pandas as pd
import os

from ..visualize import display_image, display_list

'''
    python -m osprey.eval.vsr.visualize_vsr
'''

IMG_DIR = '/mmfs1/gscratch/raivn/jspark96/data/images/coco'

segm_img_dir = '../unified-sg/unified_sg/regions/vsr/images/zeroshot_test/whole'
# segm_img_dir = '../unified-sg/unified_sg/regions/vsr/images/zeroshot_test/coco'

input_file = 'osprey/eval/results/vsr/stage3_gqa_cot_sg_relevant_left_right_v3-latest-temp0.2_top1.0.jsonl'
# input_file = 'osprey/eval/results/vsr/coco_regions/gqa_cot_sg_no_relation_stage3-latest-temp0.2_top1.0.jsonl'
output_file = 'osprey/eval/vsr/vsr_visualize.html' # usually fixed..


df = pd.read_json(input_file, lines=True)
aggregations = {col: 'first' if col == 'regions' or 'image' in col else list for col in df.columns}
df['output'] = df.apply(lambda x: f"Q: {x['question']}\n\nPred:\n{x['output']}\n\nGT: {x['answer']}", axis=1)
image_df = df.groupby('image_id', as_index=False).agg(aggregations).iloc[:400]
image_df['image'] = image_df['image_id'].apply(lambda x: os.path.join(IMG_DIR, x)).apply(display_image)
image_df['segm_image'] = image_df['image_id'].apply(lambda x: os.path.join(segm_img_dir, os.path.basename(x))).apply(display_image)
image_df = image_df[~image_df['segm_image'].isnull()]
for k in ['question', 'answer', 'output']:
    image_df[k] = image_df[k].apply(display_list)

image_df[['image', 'segm_image', 'output']].to_html(output_file, escape=False)
