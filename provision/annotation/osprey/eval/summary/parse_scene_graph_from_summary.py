''' 
Util for parsing scene graph from generated summary.
Uses pre-trained LM to extract the scene graph as JSON.
'''

import os
import glob
import math
import json
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

NOTE= "Make sure to include only numbered IDs for your subject and object."

class SceneGraphParser:
    def __init__(self, model_name: str = 'mistralai/Mixtral-8x7B-Instruct-v0.1', example_dir: str = None):
        # specify how to quantize the model
        quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
        )
        self.model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.examples = self.load_examples(example_dir)
    
    def load_examples(self, example_dir):
        prompt = open(os.path.join(example_dir, 'prompt.txt')).read()
        examples = [{"role": "user", "content": 'instruction: ' + prompt}, {"role": "assistant", "content": "OK, my task is to generate comprehensive list of relations in [subject_id, object_id, relation_name]."}]

        sg_files = glob.glob(os.path.join(example_dir, "sg_*.txt"))
        summary_files = glob.glob(os.path.join(example_dir, "summary_*.txt"))

        # Extract suffixes and create a mapping for sg files
        sg_mapping = {os.path.splitext(os.path.basename(f))[0].split('_')[1]: f for f in sg_files}

        # Extract suffixes and create a mapping for summary files
        summary_mapping = {os.path.splitext(os.path.basename(f))[0].split('_')[1]: f for f in summary_files}

        # Now pair them based on matching suffixes
        paired_files = [(sg_mapping[suffix], summary_mapping[suffix]) for suffix in sg_mapping if suffix in summary_mapping]

        # Example of loading the content of each file pair
        for sg_file, summary_file in paired_files:
            with open(sg_file, 'r') as sg_f, open(summary_file, 'r') as summary_f:
                sg_content = sg_f.read()
                summary_content = summary_f.read()
                examples.append({"role": "user", "content": summary_content,})
                examples.append({"role": "assistant", "content": sg_content,})
        
        return examples
    
    def get_examples(self):
        return self.examples
        
    def parse(self, text):

        examples = self.get_examples()

        messages = examples + [
            {"role": "user", "content": text + ' ' + NOTE}
        ]

        model_inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

        generated_ids = self.model.generate(model_inputs, max_new_tokens=1024, do_sample=False, temperature=0.2)

        generated_output = self.tokenizer.batch_decode(generated_ids[:, model_inputs.size(1):], skip_special_tokens=True)[0]

        return generated_output
    
if __name__ == '__main__':
    from tqdm import tqdm

    df = pd.read_json('data/relation/train_coco_summary_sam_seem_regions_150.jsonl',lines=True)
    data = df.to_dict(orient='records')

    n = int(os.environ.get('NUM_CHUNKS', 1))
    k = int(os.environ.get('CHUNK_IDX', 0))
    print('num_chunks: {}, chunk_idx: {}'.format(n, k))
    result = []
    chunk = get_chunk(data, n, k)

    # Load Model
    output_file = 'data/relation/train_coco_summary_sam_seem_regions_150_parsed_{}_{}.json'.format(n, k)
    mistral_parser = SceneGraphParser(example_dir='osprey/eval/relation/examples/parsing/') 
    for datum in tqdm(chunk[:4]):
        sg = mistral_parser.parse(datum['summary'])
        result.append({'image_id': datum['image_id'], 'relations': sg, 'summary': datum['summary']})
    breakpoint()
    pd.DataFrame(result).to_json(output_file, orient='records')
