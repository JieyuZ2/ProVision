from dataclasses import dataclass
from typing import Optional
import torch
import torch.utils
import torch.utils.data
import transformers
from torch.utils.data import ConcatDataset
import bisect
import os
import json
from osprey.constants import IGNORE_INDEX

from .stage2_data import COCODataset, RefCOCO, RefCOCOP
from .vcr import VCRDataset
from .vg import VGDATA
from .visualcomet import VisualCometDataset
from .stage2_data import PascalPart
from .stage2_data import PartImagenet
from .osprey_724k import OspreyDetailedDescription, OspreyConversations, OspreyShortForm, OspreyPartLevel, OspreyLVISPosNeg

from .sg import SGDataset
from .sg_detection import SGDetectionDataset
from .psg import PSGDataset
from .cot_sg import GQACoTSGDataset, GQACoTGroundedSGDataset
from .aokvqa import AOKVQADataset

from .llava import LlavaDataset
from .all_seeing import AllSeeingDataset
from .grit import GritDataset

from .refexp import RefExpDataset
from .relation_category import RelationCategoryDataset, RelationDescriptionDataset, RelationSummaryDataset
from .relation_specialist import RelationSpecialistDataset
from .point_qa import PointQALocalDataset, PointQATwiceDataset, V7WPointQADataset
from .flickr import FlickrEntitiesDataset


@dataclass
class DataCollatorForDetDataset(object):

    tokenizer: transformers.PreTrainedTokenizer
    image_token_len: int # 1024
    def __call__(self, instances):

        input_ids, labels, img_metas, masks = tuple([instance.get(key,None) for instance in instances]
                                  for key in ('input_ids',
                                              'labels',
                                              'img_metas',
                                              'masks'))
        
        # max_allowed_len_before_image = self.tokenizer.model_max_length - self.image_token_len
        # # Cut off last if the adding image sequence exceeds the maximum length
        # print('max_allowed_len_before_image:', max_allowed_len_before_image)
        # print('before input len:', [len(input_id) for input_id in input_ids])
        # print('before label len:', [len(label) for label in labels])
        # input_ids = [input_id[:max_allowed_len_before_image] if 'image' in instance else input_id
        #              for instance, input_id in zip(instances, input_ids)]
        # labels = [label[:max_allowed_len_before_image] if 'image' in instance else label
        #              for instance, label in zip(instances, labels)]
        # print('after len:', [len(input_id) for input_id in input_ids])
        # print('after lable len:', [len(label) for label in labels])
            
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            img_metas=img_metas,
            masks = masks
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch

def expand_env_vars_in_config(config):
    """
    Expands environment variables found in string values of a given dictionary.

    Parameters:
    - config (dict): The dictionary whose string values may contain environment variables.

    Returns:
    - dict: A dictionary with environment variables in its string values expanded.
    """
    def expand_value(value):
        if isinstance(value, str):
            return os.path.expandvars(value)
        elif isinstance(value, dict):
            return {k: expand_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [expand_value(item) for item in value]
        else:
            return value

    return expand_value(config)

def make_multitask_data_module(tokenizer,
                                data_args) :
    """Make dataset and collator for supervised fine-tuning."""

    if data_args.dataset_config is not None:
        dataset_config = json.load(open(data_args.dataset_config))

    train_dataset = build_osprey_dataset(dataset_config,
                            tokenizer=tokenizer,
                            data_args=data_args)

    
    assert tokenizer.model_max_length > data_args.image_token_len, 'model_max_length should be greater than image_token_len'
    data_collator = DataCollatorForDetDataset(tokenizer=tokenizer, image_token_len=data_args.image_token_len)

    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def build_osprey_dataset(dataset_config,
                  tokenizer=None,
                  data_args=None,
                  **kwargs):
    if isinstance(dataset_config, list):
        datasets = []
        for cfg in dataset_config:
            temp_dataset = build_osprey_dataset(cfg, tokenizer=tokenizer, data_args=data_args, **kwargs)
            dataset_name = temp_dataset.__class__.__name__
            datasets.append((dataset_name, temp_dataset))

        for dataset_name, dataset in datasets:
            print(type(dataset), f'len = {len(dataset)}')

        return ConcatDataset(datasets)
    
    dataset_type = dataset_config.pop('type')
    dataset_config = expand_env_vars_in_config(dataset_config)
    print('Dataset: {}'.format(dataset_type))
    print(dataset_config)

    if dataset_type == 'coco_data':
        dataset = COCODataset(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )
    
    elif dataset_type == 'vcr':
        dataset = VCRDataset(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )
    elif dataset_type == 'visualcomet':
        dataset = VisualCometDataset(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )
    elif dataset_type == 'VGDATA':
        dataset = VGDATA(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )
    elif dataset_type == 'RefCOCO':
        dataset = RefCOCO(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )
    elif dataset_type == 'RefCOCOP':
        dataset = RefCOCOP(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )
    elif dataset_type == 'PascalPart':
        dataset = PascalPart(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )
    elif dataset_type == 'PartImagenet':
        dataset = PartImagenet(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )
    elif dataset_type == 'OspreyDetailedDescription':
        dataset = OspreyDetailedDescription(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )
    elif dataset_type == 'OspreyConversations':
        dataset = OspreyConversations(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )
    elif dataset_type == 'OspreyShortForm':
        dataset = OspreyShortForm(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )
    elif dataset_type == 'OspreyPartLevel':
        dataset = OspreyPartLevel(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )
    elif dataset_type == 'OspreyLVISPosNeg':
        dataset = OspreyLVISPosNeg(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )
    
    ## pre-train images
    elif dataset_type == 'llava':
        dataset = LlavaDataset(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )
    elif dataset_type == 'all_seeing':
        dataset = AllSeeingDataset(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )
    elif dataset_type == 'grit':
        dataset = GritDataset(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )
    ### 

    elif dataset_type == 'sg':
        dataset = SGDataset(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )
    elif dataset_type == 'sg_detection':
        dataset = SGDetectionDataset(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )
    elif dataset_type == 'psg':
        dataset = PSGDataset(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )
    
    elif dataset_type == 'point_qa_local':
        dataset = PointQALocalDataset(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )

    elif dataset_type == 'point_qa_twice':
        dataset = PointQATwiceDataset(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )

    elif dataset_type == 'v7w_point_qa':
        dataset = V7WPointQADataset(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )
    elif dataset_type == 'flickr_entities':
        dataset = FlickrEntitiesDataset(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )

    elif dataset_type == 'refexp':
        dataset = RefExpDataset(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )

    elif dataset_type == 'relation':
        dataset = RelationCategoryDataset(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )
    elif dataset_type == 'relation_description':
        dataset = RelationDescriptionDataset(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )
    elif dataset_type == 'relation_summary':
        dataset = RelationSummaryDataset(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )
    
    elif dataset_type == 'relation_specialist':
        dataset = RelationSpecialistDataset(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )
        
    elif dataset_type == 'gqa_cot_sg':
        dataset = GQACoTSGDataset(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )
    elif dataset_type == 'gqa_cot_sg_grounded':
        dataset = GQACoTGroundedSGDataset(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )
    elif dataset_type == 'aokvqa':
        dataset = AOKVQADataset(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )

    else:
        raise NotImplementedError

    return dataset



class ConcatDataset(ConcatDataset):
    def __init__(self, datasets: list[tuple[str, torch.utils.data.Dataset]]):
        """ datasets: [(name, dataset), ...] """
        dataset_names, datasets =  zip(*datasets)
        self.dataset_names = dataset_names
        super().__init__(datasets)
    
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        
        dataset: torch.utils.data.Dataset = self.datasets[dataset_idx]
        dataset_name: str = self.dataset_names[dataset_idx]

        item = dataset[sample_idx]
        item['source'] = dataset_name

        return item
    

    def collater(self, samples):

        all_keys = set()
        for s in samples:
            all_keys.update(s)

        shared_keys = all_keys
        for s in samples:
            shared_keys = shared_keys & set(s.keys())

        samples_shared_keys = []
        for s in samples:
            samples_shared_keys.append({k: s[k] for k in s.keys() if k in shared_keys})

        return self.datasets[0].collater(samples_shared_keys)

if __name__ == '__main__':
    from transformers import AutoTokenizer, CLIPImageProcessor
    from osprey.train.train import DataArguments
    from osprey.model.multimodal_encoder.clip_encoder import get_image_processor
    from tqdm import tqdm 
    import sys

    tokenizer = AutoTokenizer.from_pretrained('/net/nfs.cirrascale/mosaic/jamesp/models/Osprey-7b/')
    image_processor = get_image_processor(img_size=512)
    data_args = DataArguments(image_aspect_ratio='pad')
    data_args.mm_use_im_start_end = False
    data_args.image_processor = image_processor
    data_args.is_multimodal = True
    if len(sys.argv) > 1:
        data_args.dataset_config = sys.argv[1]
    else:
        data_args.dataset_config = 'osprey/configs/finetune/multi_region_v10_e2e_sg.json'
    print('dataset config:', data_args.dataset_config)

    dataset = make_multitask_data_module(
        tokenizer,
        data_args
    )
    dataset = dataset['train_dataset']
    input_ids = dataset[0]['input_ids']
    input_ids = input_ids[input_ids >= 0]
    print(tokenizer.decode(input_ids))
    breakpoint()

    bad_instance = []
    for idx in tqdm(range(len(dataset))):
        try:
            dataset[idx]
        except Exception:
            print('found bad instance: {}'.format(idx))
            bad_instance.append(idx)
    breakpoint()
            
    

