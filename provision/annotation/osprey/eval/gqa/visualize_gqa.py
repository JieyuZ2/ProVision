import pandas as pd
import os
import base64

''' python osprey/eval/gqa/visualize_gqa.py '''
IMAGE_DIR = '/mmfs1/gscratch/raivn/jspark96/data/images/gqa/images'
ONLY_INCORRECT = False

input_file = 'osprey/eval/results/gqa/testdev/stage3_gqa_cot_sg_relevant_left_right_v3-latest-temp0.2_top1.0.jsonl'
input_file = 'osprey/eval/results/gqa/val_balanced_aokvqa/gqa_regions/gqa_cot_sg_relevant_left_right_v3_stage3_sam_hq-latest-temp0.2_top0.95.jsonl'
input_file = 'osprey/eval/results/gqa/val_balanced_aokvqa/gqa_regions/vg_sg_gqa_sg_cot_v4_sam_hq-latest-temp0.2_top0.95.jsonl'
input_file = 'osprey/eval/results/gqa/testdev/default_sam_regions/vg_sg_gqa_sg_cot_v4-sam_hq-latest-temp0.2_top1.0.jsonl'
output_file = 'osprey/eval/gqa/gqa_visualize.html'
if ONLY_INCORRECT:
    output_file = output_file.replace('.html', '_incorrect.html')

segm_img_dir = '../unified-sg/unified_sg/regions/gqa/images/testdev/whole'
# segm_img_dir = '../unified-sg/unified_sg/regions/gqa/images/val_balanced_aokvqa/gqa_regions'

def display_list(output):
    return '<br><br>==<br>'.join(output).replace('\n','<br>')

def display_image(image_path, width=600, height=600):
    if not os.path.isfile(image_path):
        return None
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return f'<img src="data:image/png;base64,{encoded_string}" width="{width}" height="{height}" />'  # Adjust width and format as needed

# def display_image(image_path, width=200, height=200):
#     return f'<img src="{image_path}" width="{width}" height="{height}" />'  # Adjust width as needed


df = pd.read_json(input_file, lines=True)
if ONLY_INCORRECT:
    df = df[df['pred_answer'] != df['answer']]
aggregations = {col: 'first' if col == 'regions' or 'image' in col else (lambda x: list(x)[:5]) for col in df.columns}
df['output'] = df.apply(lambda x: f"Q: {x['question']}\n\nPred: {x['output']}\n\nGT: {x['answer']}", axis=1)
image_df = df.groupby('image_id', as_index=False).agg(aggregations).iloc[:500]
image_df['image'] = image_df['image_id'].apply(lambda x: os.path.join(IMAGE_DIR, str(x)+'.jpg')).apply(display_image)
image_df['segm_image'] = image_df['image_id'].apply(lambda x: os.path.join(segm_img_dir, str(x)+'.jpg')).apply(display_image)
image_df = image_df[~image_df['segm_image'].isnull()]
for k in ['question', 'answer', 'output']:
    image_df[k] = image_df[k].apply(display_list)

image_df[['image_id', 'image', 'segm_image', 'output']].to_html(output_file, escape=False)
