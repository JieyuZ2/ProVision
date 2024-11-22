import string
import warnings
from typing import Dict, List

from .mcq_template import make_multiple_choice_qa
from .object_annotation import OBJECT_FOR_COUNTING_TASK


def check_object_for_counting_task(object_name):
	if object_name not in OBJECT_FOR_COUNTING_TASK:
		warnings.warn(f"{object_name} is not in OBJECT_FOR_COUNTING_TASK, default is NOT. Please add this object to object_annotation.py")
	return OBJECT_FOR_COUNTING_TASK.get(object_name, 0)


def get_cnt_word_(cnt, inflect_engine, number_mode):
	if number_mode == 'numeric':
		cnt_word = str(cnt)
	elif number_mode == 'word':
		cnt_word = inflect_engine.number_to_words(cnt)
	else:
		raise ValueError(f"number_mode should be 'numeric' or 'word', but got {number_mode}")
	return cnt_word


def get_cnt_word(cnt, rng, inflect_engine, number_mode):
	if number_mode not in ['numeric', 'word']:
		if number_mode != 'random':
			warnings.warn(f"number_mode should be 'numeric' or 'word', but got {number_mode}. Use random mode.")
		number_mode = rng.choice(['numeric', 'word'])
	if isinstance(cnt, list):
		return [get_cnt_word_(c, inflect_engine, number_mode) for c in cnt]
	else:
		return get_cnt_word_(cnt, inflect_engine, number_mode)


def bbox_coordinate_to_ratio(bbox, height, width, digits=2):
	"""
	Convert bbox coordinates to ratio.
	"""
	x1, y1, x2, y2 = bbox
	return float(round(x1 / width, digits)), float(round(y1 / height, digits)), float(round(x2 / width, digits)), float(round(y2 / height, digits))


def point_coordinate_to_ratio(point, height, width, digits=2):
	"""
	Convert point coordinates to ratio.
	"""
	x, y = point
	return float(round(x / width, digits)), float(round(y / height, digits))


def cut_bbox(bbox, image):
	"""
	Cut the bounding box from the image.
	"""
	x1, y1, x2, y2 = bbox
	return image[y1:y2, x1:x2]


def safe_sample(rng, candidates, n, exclude=[]):
	if len(exclude) > 0:
		candidates = [x for x in candidates if x not in exclude]
	if len(candidates) <= n:
		return candidates
	return list(rng.choice(candidates, n, replace=False))


def make_and_description(names, rng=None):
	if not isinstance(names, list):
		names = list(names)

	if len(names) == 0:
		return ""
	if len(names) == 1:
		return names[0]

	if rng is not None:
		names = list(rng.permutation(names))

	if len(names) == 2:
		return ' and '.join(names)
	return ', '.join(names[:-1] + [f'and {names[-1]}'])


def _fill_formatted_string(data_info, formatted_string, rng):
	key_words = [tup[1] for tup in string.Formatter().parse(formatted_string) if tup[1] is not None]
	contained_kwargs = {}
	for k, v in data_info.items():
		if k in key_words:
			if isinstance(v, list):
				contained_kwargs[k] = make_and_description(v, rng)
			else:
				contained_kwargs[k] = v
	# fill in the formatted string
	return formatted_string.format(**contained_kwargs)


def make_mqa(data, data_info, rng):
	assert 'prompt' in data and 'response' in data, "prompt and response should be in the data"
	assert 'candidates' in data_info and 'answer' in data_info, "candidates and answer should be in the data_info"
	return make_multiple_choice_qa(data, data_info['candidates'], data_info['answer'], rng)


def _make_data_helper(data_info: Dict, template: Dict, rng, multiple_choice) -> Dict:
	# fill in the template
	data = {k: _fill_formatted_string(data_info, item, rng) for k, item in template.items()}
	if multiple_choice:
		data = make_mqa(data, data_info, rng)
	if 'metadata' in data_info:
		data['metadata'] = data_info['metadata']
	return data


def make_one_data(data_info: Dict, templates: List, rng, enumerate_templates: bool = True, multiple_choice: bool = False) -> List:
	# initialize a random number generator
	if enumerate_templates:
		instruction = [_make_data_helper(data_info, template, rng, multiple_choice) for template in templates]
	else:
		# choose a random template
		template = rng.choice(templates)
		instruction = [_make_data_helper(data_info, template, rng, multiple_choice)]

	return instruction


def make_data(data_info_list: List[Dict], templates: List, rng, enumerate_templates: bool = True) -> List:
	# initialize a random number generator
	data_list = []
	for data_info in data_info_list:
		data_list += make_one_data(data_info, templates, rng, enumerate_templates)
	return data_list
