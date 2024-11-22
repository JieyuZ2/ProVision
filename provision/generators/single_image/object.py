from collections import Counter
from typing import List

import inflect

from ..template import get_qa_template
from ..utils import check_object_for_counting_task, get_cnt_word
from ...base import BaseGenerator
from ...dataset import BoundBoxes, JointDataset


class ExistsObjectGenerator(BaseGenerator):
	"""
	"prompt"  : "How many {name}?",
	"response": "{name}"
	"""

	qa_templates = get_qa_template("ExistsObjectGenerator")
	des_templates = [
		{
			"description": "There {be} {count_name}."
		}
	]
	inflect_engine = inflect.engine()
	metadata: bool = True

	def __init__(self, dataset: JointDataset, number_mode="random", **kwargs):
		super().__init__(dataset=dataset, **kwargs)
		self.number_mode = number_mode

	def _generate(self, annotation: BoundBoxes) -> List:
		data_list = []

		for name, cnt in Counter(annotation.labels).items():
			if cnt == 1:
				candidates = [1, 2, 3, 4]
			else:
				candidates = [cnt, cnt - 1, cnt + 1, cnt + 2]
			candidates = get_cnt_word(candidates, self.rng, self.inflect_engine, self.number_mode)
			cnt_word = candidates[0]

			be = 'is'
			if check_object_for_counting_task(name):
				if cnt > 1:
					be = 'are'
					name = self.inflect_engine.plural(name)
				count_name = cnt_word + ' ' + name
			else:
				if self.template_mode == 'qa':
					continue
				be = 'are'
				cnt_word = ''
				count_name = name

			data_list += self.make_one_data(
				{
					"name"      : name,
					"count_name": count_name,
					"be"        : be,
					"count"     : cnt_word,
					"candidates": candidates,
					"answer"    : cnt_word,
					"metadata"  : {
						"object": [name],
						"count" : [cnt]
					}
				},
			)
		return data_list


class MostObjectGenerator(BaseGenerator):
	"""
	"prompt"  : "Among {candidates}, which is the most frequent object?",
	"response": "{name}"
	"""

	qa_templates = get_qa_template("MostObjectGenerator")
	des_templates = [
		{
			"description": "Among {candidates}, {name} is the most frequent object."
		}
	]
	metadata: bool = True

	def _generate(self, annotation: BoundBoxes) -> List:
		annotation = annotation.non_including_bboxes()

		labels = [label for label in annotation.labels if check_object_for_counting_task(label)]

		cnt_dict = Counter(labels)
		if len(cnt_dict) < 2:
			return []
		name = max(cnt_dict, key=cnt_dict.get)
		max_cnt = cnt_dict[name]

		labels = [i for i in set(labels) if cnt_dict[i] < max_cnt]
		if len(labels) == 0:
			return []
		if len(labels) > self.n_choice - 1:
			candidates = [name] + list(self.rng.choice(labels, self.n_choice - 1, replace=False))
		else:
			candidates = [name] + labels
		return self.make_one_data(
			{
				"name"      : name,
				"candidates": candidates,
				"answer"    : name,
				"metadata"  : {
					"object": [name]
				}
			},
		)


class LeastObjectGenerator(BaseGenerator):
	"""
	"prompt"  : "Among {candidates}, which object appears the least?",
	"response": "{name}"
	"""

	qa_templates = get_qa_template("LeastObjectGenerator")
	des_templates = [
		{
			"description": "Among {candidates}, {name} is the least frequent object."
		}
	]
	metadata: bool = True

	def _generate(self, annotation: BoundBoxes) -> List:
		annotation = annotation.non_including_bboxes()

		labels = [label for label in annotation.labels if check_object_for_counting_task(label)]

		cnt_dict = Counter(labels)
		if len(cnt_dict) < 2:
			return []
		name = min(cnt_dict, key=cnt_dict.get)
		min_cnt = cnt_dict[name]

		labels = [i for i in set(labels) if cnt_dict[i] > min_cnt]
		if len(labels) == 0:
			return []
		if len(labels) > self.n_choice - 1:
			candidates = [name] + list(self.rng.choice(labels, self.n_choice - 1, replace=False))
		else:
			candidates = [name] + labels
		return self.make_one_data(
			{
				"name"      : name,
				"candidates": candidates,
				"answer"    : name,
				"metadata"  : {
					"object": [name]
				}
			},
		)


class LeftMostObjectGenerator(BaseGenerator):
	"""
	"prompt"  : "Among {candidates}, which is on the most left side?",
	"response": "{name}"
	"""

	qa_templates = get_qa_template("LeftMostObjectGenerator")
	des_templates = [
		{
			"description": "Among {candidates}, {name} is on the most left side."
		}
	]
	metadata: bool = True

	def __init__(self, dataset: JointDataset, n=3, **kwargs):
		super().__init__(dataset=dataset, **kwargs)
		self.n_choice = n

	def _generate(self, annotation: BoundBoxes) -> List:
		n_non_overlapping_bboxes_list = annotation.n_non_overlapping_bboxes(self.n_choice, horizontal=True, unique=True)

		data_list = []
		for n_non_overlapping_bboxes in n_non_overlapping_bboxes_list:
			ann = annotation.subset(n_non_overlapping_bboxes)
			bboxes_with_idx = [(bbox, i) for i, bbox in enumerate(ann.bboxes)]
			name = ann.labels[min(bboxes_with_idx, key=lambda x: x[0][0])[1]]

			data_list += self.make_one_data(
				{
					"name"      : name,
					"candidates": ann.labels,
					"answer"    : name,
					"metadata"  : {
						"object": [name]
					}
				},
			)

		return data_list


class RightMostObjectGenerator(BaseGenerator):
	"""
	"prompt"  : "Among {candidates}, which is on the most right side?",
	"response": "{name}"
	"""

	qa_templates = get_qa_template("RightMostObjectGenerator")
	des_templates = [
		{
			"description": "Among {candidates}, {name} is on the most right side."
		}
	]
	metadata: bool = True

	def __init__(self, dataset: JointDataset, n=3, **kwargs):
		super().__init__(dataset=dataset, **kwargs)
		self.n_choice = n

	def _generate(self, annotation: BoundBoxes) -> List:
		n_non_overlapping_bboxes_list = annotation.n_non_overlapping_bboxes(self.n_choice, horizontal=True, unique=True)

		data_list = []
		for n_non_overlapping_bboxes in n_non_overlapping_bboxes_list:
			ann = annotation.subset(n_non_overlapping_bboxes)
			bboxes_with_idx = [(bbox, i) for i, bbox in enumerate(ann.bboxes)]
			name = ann.labels[max(bboxes_with_idx, key=lambda x: x[0][2])[1]]

			data_list += self.make_one_data(
				{
					"name"      : name,
					"candidates": ann.labels,
					"answer"    : name,
					"metadata"  : {
						"object": [name]
					}
				},
			)

		return data_list


class TopMostObjectGenerator(BaseGenerator):
	"""
	"prompt"  : "Among {candidates}, which is on the most top side?",
	"response": "{name}"
	"""

	qa_templates = get_qa_template("TopMostObjectGenerator")
	des_templates = [
		{
			"description": "Among {candidates}, {name} is on the most top side."
		}
	]
	metadata: bool = True

	def __init__(self, dataset: JointDataset, n=3, **kwargs):
		super().__init__(dataset=dataset, **kwargs)
		self.n_choice = n

	def _generate(self, annotation: BoundBoxes) -> List:
		n_non_overlapping_bboxes_list = annotation.n_non_overlapping_bboxes(self.n_choice, vertical=True, unique=True)

		data_list = []
		for n_non_overlapping_bboxes in n_non_overlapping_bboxes_list:
			ann = annotation.subset(n_non_overlapping_bboxes)
			bboxes_with_idx = [(bbox, i) for i, bbox in enumerate(ann.bboxes)]
			name = ann.labels[min(bboxes_with_idx, key=lambda x: x[0][1])[1]]

			data_list += self.make_one_data(
				{
					"name"      : name,
					"candidates": ann.labels,
					"answer"    : name,
					"metadata"  : {
						"object": [name]
					}
				},
			)

		return data_list


class BottomMostObjectGenerator(BaseGenerator):
	"""
	"prompt"  : "Among {candidates}, which is on the most bottom side?",
	"response": "{name}"
	"""

	qa_templates = get_qa_template("BottomMostObjectGenerator")
	des_templates = [
		{
			"description": "Among {candidates}, {name} is on the most bottom side."
		}
	]
	metadata: bool = True

	def __init__(self, dataset: JointDataset, n=3, **kwargs):
		super().__init__(dataset=dataset, **kwargs)
		self.n_choice = n

	def _generate(self, annotation: BoundBoxes) -> List:
		n_non_overlapping_bboxes_list = annotation.n_non_overlapping_bboxes(self.n_choice, vertical=True, unique=True)

		data_list = []
		for n_non_overlapping_bboxes in n_non_overlapping_bboxes_list:
			ann = annotation.subset(n_non_overlapping_bboxes)
			bboxes_with_idx = [(bbox, i) for i, bbox in enumerate(ann.bboxes)]
			name = ann.labels[max(bboxes_with_idx, key=lambda x: x[0][3])[1]]

			data_list += self.make_one_data(
				{
					"name"      : name,
					"candidates": ann.labels,
					"answer"    : name,
					"metadata"  : {
						"object": [name]
					}
				},
			)

		return data_list


ObjectGeneratorList = [
	ExistsObjectGenerator,
	MostObjectGenerator,
	LeastObjectGenerator,
	LeftMostObjectGenerator,
	RightMostObjectGenerator,
	TopMostObjectGenerator,
	BottomMostObjectGenerator
]
