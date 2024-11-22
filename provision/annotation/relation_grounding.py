import json
import re
from typing import Union

import torch
from tqdm import tqdm


def rel_parsing(
		rel_output: Union[str, list],
		detection_mode: str = "holistic"
):
	img_relations = []
	if detection_mode == 'holistic':
		# keep only relations
		try:
			outputs_list = rel_output.strip().split('Relations:')[1].split('\n')[1:]  # ['regionX: regionY RELATION, regionZ RELATION...', ...] e.g., region2: region4 on
		except:
			return []

		for output in outputs_list:
			try:
				region_numbers = re.findall(r'region(\d+)', output)
				target_idx = int(region_numbers[0])
				source_idx = list(map(int, region_numbers[1:]))
				relations = re.sub(r'region(\d+)', '', output).strip().split(':')[-1].strip().split(',')

				# the "idx" is in the order of bboxes and labels
				for idx_rel in zip(source_idx, relations):
					img_relations.append((idx_rel[0]-1, idx_rel[1].strip(), target_idx-1))  # Y, relation, X
			except:
				continue

	elif detection_mode == 'detailed':
		for idx, rel_output in enumerate(rel_output):
			rel_list = rel_output.split(', ')
			img_relations += [(int(re.findall(r'region(\d+)', rel)[0]) - 1,
							   re.sub(r'region(\d+)', '', rel).strip(),
							   idx) for rel in rel_list]  # (id, relation, current_idx)

	else:
		raise f"Invalid detection mode {detection_mode}"

	return list(set(img_relations))


def cate_grounding(output_list: list,
				   categlories_list_path: str,
				   device: str = "cuda",
				   batch_size: int = 10000,
				   st_model_name: str = "all-mpnet-base-v2",
				   top_k: int = 1):
	from sentence_transformers import SentenceTransformer
	model = SentenceTransformer(st_model_name).to(device)

	cate_dict = json.load(open(categlories_list_path, 'r'))
	cate_list = []
	for sub_cate_list in cate_dict.values():
		cate_list += sub_cate_list

	cate_embedding = model.encode(cate_list, batch_size=batch_size, convert_to_tensor=True, normalize_embeddings=True)  # len(cate_list), 768

	res = []
	bp = []
	relation_list = []
	for ol in tqdm(output_list, desc="relation grounding"):
		relation_list += [triplet[1] for triplet in ol]
		bp.append((len(relation_list), ol))
		if len(relation_list) > batch_size:
			relation_embedding = model.encode(relation_list, batch_size=batch_size, convert_to_tensor=True, normalize_embeddings=True)  # len(relation_list), 768
			selected_cate_idx = torch.topk(relation_embedding @ cate_embedding.T, top_k, dim=1).indices.cpu().numpy()

			cur = 0
			for b, o in bp:
				new_relation_list = [(o[i][0], cate_list[idx], o[i][2])
									 for i, idx_list in enumerate(selected_cate_idx[cur:b]) for idx in idx_list]
				res.append(new_relation_list)
				cur = b

			relation_list = []
			bp = []

	if len(bp) > 0:
		relation_embedding = model.encode(relation_list, batch_size=batch_size, convert_to_tensor=True, normalize_embeddings=True)  # len(relation_list), 768
		selected_cate_idx = torch.topk(relation_embedding @ cate_embedding.T, top_k, dim=1).indices.cpu().numpy()

		cur = 0
		for b, o in bp:
			new_relation_list = [(o[i][0], cate_list[idx], o[i][2])
								 for i, idx_list in enumerate(selected_cate_idx[cur:b]) for idx in idx_list]
			res.append(new_relation_list)
			cur = b

	return [{
		"relations": r
	} for r in res]
