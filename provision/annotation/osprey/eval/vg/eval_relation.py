import argparse
from pathlib import Path
import torch
import os
import json
from tqdm import tqdm
import pandas as pd
import re
from functools import partial
from typing import List, Literal, Dict, Tuple
from transformers import AutoTokenizer, CLIPImageProcessor
from osprey.constants import IMAGE_TOKEN_INDEX
from osprey.conversation import conv_templates, SeparatorStyle
from osprey.mm_utils import tokenizer_image_token
from osprey.train.train import preprocess_multimodal
from osprey.train.train import DataArguments
from osprey.model.language_model.osprey_llama import OspreyLlamaForCausalLM

from osprey.datasets.relation import RelationDataset, RELATION_QUESTIONS, Ref_WAY
from osprey.eval.eval import OspreyEval

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import argparse

data_args = DataArguments()
data_args.mm_use_im_start_end = False
data_args.is_multimodal = True

class RelationEval(OspreyEval, RelationDataset):
    ''' 
    Scene Graph evaluation class that assigns generation result to scene graph object and relationship class
    using BertModel.
    '''
    def __init__(
            self, 
            model_path, 
            bert_model, 
            category,
            max_regions=150,
            region_mode: Literal['box', 'segmentation', 'box_segmentation']='segmentation',
            chunk_idx:int=0,
            num_chunks:int=1,
            sg_mode: int=3, 
            debug=False, ):
        
        super().__init__(model_path, chunk_idx=chunk_idx, num_chunks=num_chunks, debug=debug)
        self.max_regions = max_regions
        self.sg_mode = sg_mode

        self.region_mode = region_mode
        self.use_box = region_mode == 'box'
        self.is_train = False

        # Set object and relationship labels
        category_dicts = json.load(open(category))

        # Add background 
        category_dicts['idx_to_label'][0] = 'background'
        category_dicts['idx_to_predicate'][0] = 'none'

        self.idx_to_label: Dict[int, str] = {int(k): v for k,v in category_dicts['idx_to_label'].items()}
        self.label_to_idx: Dict[str, int] = {v: k for k,v in category_dicts['idx_to_label'].items()}
        self.idx_to_predicate: Dict[int, str] = {int(k): v for k,v in category_dicts['idx_to_predicate'].items()}
        self.predicate_to_idx: Dict[str, int] = {v: k for k,v in category_dicts['idx_to_predicate'].items()}

        self.idx_to_attribute: Dict[int, str] = {int(k): v for k,v in category_dicts['idx_to_attribute'].items()}

        # Set Object Embeddings
        self.bert_model = SentenceTransformer(bert_model) 
        object_categories = [self.idx_to_label[k] for k in sorted(self.idx_to_label)]
        self.object_embeddings = self.bert_model.encode(object_categories, convert_to_tensor=True)

    def eval(self, root_path, ann_file, temperature=0.2, top_p=1.0):

        sgs = []
        anns = pd.read_json(ann_file, lines=True).to_dict(orient='records')

        for ann in tqdm(anns[:100]):
            image_id = str(ann['image_id'])
            height = ann['height']
            width = ann['width']
            regions: list = ann['regions'][:self.max_regions]

            # Gather regions
            region_sg = {}
            segs = []
            boxes = []
            for idx,region in enumerate(regions):
                ''' index 0 <-> region1'''

                name = self.idx_to_label[region['labels']]
                attributes = [self.idx_to_attribute[att] for att in region['attributes'] if att != 0]
                relations = [f"{self.idx_to_predicate[rel]} region{r+1}" for r, rel in enumerate(region['relation']) if rel != 0]
                
                region_id = idx + 1
                sg_dict = {'name': name, 'attributes': attributes, 'relations': relations}
                region_sg[region_id] = sg_dict

                # add region segmentations
                boxes.append(region['bbox'])
                segs.append(region['segmentation'])
            
            img_path = os.path.join(root_path, image_id+'.jpg')

            # Add region prompts
            num_objects = len(segs)
            ref_string = ''
            for i in range(num_objects):
                ref_string = ref_string +  f'region{i+1} <mask><pos>' + ','
            ref_string = ref_string[:-1]
            ref_prefix = Ref_WAY[0]

            begin_string = ref_prefix.replace('<region>', ref_string)

            # GT scene graph for reference
            gt = self.create_conversation(region_sg, begin_string)
            gt_object = gt[1]['value']
            gt_relation = gt[3]['value']

            # Create masks and regions
            masks = self.create_masks(boxes, segs, height, width)
            masks = torch.from_numpy(masks)

            # Generate relations for each region.
            region_outputs = []
            for id in range(len(regions)):
                prompt = begin_string + ' ' + RELATION_QUESTIONS[0]

                subj_region = 'region' + str(id+1)
                prompt = prompt.format(subj_region)
                init_inputs = self.get_init_inputs(img_path,
                                            self.image_processor,
                                            masks=masks,
                                            prompt=prompt,
                                            )

                image = init_inputs['image']
                masks = init_inputs['masks'].cuda()

                conv = self.get_new_conv()
                qs = init_inputs['sources'][0][0]['value']

                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

                # Generated relations per object
                # 'region4 in front of, region2 to the left of, region3 behind, region5 above'
                outputs: str = self.get_outputs(image, input_ids, masks, self.stop_str, temperature, top_p, max_new_tokens=1024)
                outputs = f"{subj_region}: {outputs}"
                region_outputs.append(outputs)

            print("[GT]")
            print(gt_object)
            print(gt_relation)

            print("[Pred]")
            print('\n'.join(region_outputs))

            # Parse prediction
            prediction_labels: dict = self.get_prediction_labels(region_outputs, len(regions))

            sgs.append({
                'image_id': image_id,
                'region_sg': region_sg,
                'bboxes': boxes,
                'output': outputs,
                'prediction_labels': prediction_labels,
                'gt_relation': gt_relation,
                'width': width, 
                'height': height
            })

        return sgs

    def get_prediction_labels(self, relation_outputs: str, object_num: int):
        '''
        Extracts object and relation prediction labels from raw string.
        '''

        predictions = {}

        relation_triplets: List[Tuple[int,str,int]] = self.get_relation_triplets(relation_outputs)
        relation_preds: dict = self.get_relation_labels(relation_triplets, object_num)
        predictions.update(relation_preds)

        return predictions

    def get_relation_triplets(self, relations: List[str]) -> List[Tuple]:
        """
        Extract relation string into a list of triplets.
        """

        def get_triplet(relation_str: str) -> Tuple[int, str, int]:
            """
            Extracts the triplet (source, relation, target) from a relation string.

            Update [4/3/24]:
                Generated output now has the format: 'region<id> relation, ...'. 
                This is useful for getting relation calibration score for given object_id.
            """
            parts = relation_str.split(':')
            source_id: int = self.get_region_id(parts[0])

            # Find all occurrences of patterns like "region{id} relation"
            if ',' in parts[1]:
                # Handling multiple relations for the same source region
                relations = parts[1].split(',')
                for relation in relations:
                    relation = relation.strip()
                    try:
                        target_str, relation_type = relation.split(' ', 1)
                        target_id: int = self.get_region_id(target_str)
                        yield (source_id, relation_type, target_id)
                    except ValueError as e:
                        print('Failed to parse relation: {}'.format(relation))
                        continue
            else:
                target_str, relation_type = parts[1].strip().split(' ', 1)
                target_id = self.get_region_id(target_str)
                yield (source_id, relation_type, target_id)
            
        triplets = []

        for rel in relations:
            if rel.count(':') == 1:
                for triplet in get_triplet(rel):
                    triplets.append(triplet)
        return triplets

    def get_relation_labels(self, triplets: List[Tuple], object_num: int)-> Dict[str, torch.Tensor]:
        """
        Map relation triplet to relation labels using BertModel.
        """
            
        rel_pair_idxs = []
        pred_rel_scores = []
        pred_rel_labels = []
        relations = torch.zeros((object_num, object_num), dtype=torch.long)
        
        for triplet in triplets:
            subj, rel, obj = triplet
            
            if subj >= object_num or obj >= object_num:
                continue
            
            # Extract subject and object label from object_preds
            subj_text = str(subj)
            obj_text = str(obj)
            
            triplet_text = f"{subj_text} {rel} {obj_text}"
            candidates = [f"{subj_text} {r} {obj_text}" for idx, r in sorted(self.idx_to_predicate.items())]
            
            # Get closest relationship using BertModel
            cur_rel_emb = self.bert_model.encode([triplet_text], convert_to_tensor=True)
            rel_embeddings = self.bert_model.encode(candidates, convert_to_tensor=True)
            pred_logits = util.cos_sim(cur_rel_emb, rel_embeddings)
            _, pred_rel_label = pred_logits.max(dim=1)
            
            # store data
            if len(pred_logits) > 0:
                relations[subj, obj] = pred_rel_label[0]
                rel_pair_idxs.append([triplet[0], triplet[2]])
                pred_rel_scores.append(pred_logits)
                pred_rel_labels.append(pred_rel_label)
        
        # Add Dummy
        if len(pred_rel_scores) == 0:
            pred_rel_scores = [torch.zeros(1, len(self.idx_to_predicate))]
            pred_rel_labels = [torch.zeros(1, dtype=torch.long)]
        
        return {
            'rel_pair_idxs': torch.LongTensor(rel_pair_idxs),
            'pred_rel_scores': torch.cat(pred_rel_scores).detach().cpu(),
            'pred_rel_labels': torch.cat(pred_rel_labels).detach().cpu(),
            'relations': relations.detach().cpu()
        }


if __name__ == "__main__":
    '''
        python -m osprey.eval.sg.eval_relation --model exp/relation_coco_sam_seem_subset/ \
            --json data/sg/vg_test_sg_sam_hq_subset.jsonl \
            --temperature 0.5 \
            --top_p 0.9 \
            --output osprey/eval/results/relation/test/relation_coco_sam_seem_subset-gt_objects_temp0.5_top0.9.pt
    '''
    parser = argparse.ArgumentParser(description='osprey sg evaluation', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', help='path to osprey model', required=True)
    parser.add_argument('--bert', help='path to bert model', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--json', help='path to gqa json file with regions', default='data/sg/vg_test_sg_sam_hq_subset.jsonl')# default='data/sg/test_vg_sg_sam_hq.json')
    parser.add_argument('--category', help='path to VG Categories', default='osprey/eval/sg/datasets/vg/VG-SGG-dicts-with-attri.json')
    parser.add_argument('--img', help='path to gqa imgs', default='/mmfs1/gscratch/raivn/jspark96/data/images/gqa/images')
    parser.add_argument('--sg_mode', type=int, default=3)
    parser.add_argument('--region_mode', type=str, default='segmentation', choices=['box', 'segmentation', 'box_segmentation'])
    parser.add_argument('--temperature', help='temperature generation params', type=float, default=0.2)
    parser.add_argument('--top_p', help='top_p generation params', type=float, default=1.0)
    parser.add_argument('--output', help='path to save results to json file', required=True)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--use_gt_objects', action='store_true', help='used gt objects when generating relations for sg_mode: 3')
    args = parser.parse_args()

    relation_eval = RelationEval(args.model, args.bert, args.category,
                     debug=args.debug,
                     region_mode=args.region_mode,
                    )
    results = relation_eval.eval(args.img, args.json, args.temperature)
    os.makedirs(Path(args.output).parent, exist_ok=True)
    torch.save(results, args.output)

