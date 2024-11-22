from typing import List

import numpy as np

from ..template import get_qa_template
from ...base import BaseGenerator
from ...dataset import JointAnnotation


def majority_of_array(arr, mask):
	unique, counts = np.unique(arr[mask], return_counts=True)
	if len(unique) == 1:
		return unique[0]
	return unique[np.argmax(counts)]


class CompareObjectDepthGenerator(BaseGenerator):

	def __init__(self, dataset, threshold=50, **kwargs):
		super().__init__(dataset=dataset, **kwargs)
		self.threshold = threshold

	def two_object(self, annotation: JointAnnotation) -> List:
		n_non_overlapping_bboxes_list = annotation.n_non_overlapping_bboxes(n=2, unique=True)
		depth_mask = annotation.depth.mask
		seg_mask = annotation.segment.mask
		labels = annotation.labels

		data_list = []
		for (i, j) in n_non_overlapping_bboxes_list:
			seg_mask1, seg_mask2 = seg_mask[i], seg_mask[j]
			if np.sum(seg_mask1) == 0 or np.sum(seg_mask2) == 0:
				continue

			dep1, dep2 = majority_of_array(depth_mask, seg_mask1), majority_of_array(depth_mask, seg_mask2)
			if abs(float(dep1) - float(dep2)) < self.threshold:
				continue

			if dep1 < dep2:
				closer, farther = labels[j], labels[i]
			else:
				closer, farther = labels[i], labels[j]

			data_list.append((closer, farther))
		return data_list

	def three_object(self, annotation: JointAnnotation) -> List:
		n_non_overlapping_bboxes_list = annotation.n_non_overlapping_bboxes(n=3, unique=True)
		depth_mask = annotation.depth.mask
		seg_mask = annotation.segment.mask
		labels = annotation.labels

		data_list = []
		for (i, j, k) in n_non_overlapping_bboxes_list:
			seg_mask1, seg_mask2, seg_mask3 = seg_mask[i], seg_mask[j], seg_mask[k]
			if np.sum(seg_mask1) == 0 or np.sum(seg_mask2) == 0 or np.sum(seg_mask3) == 0:
				continue

			dep1, dep2, dep3 = majority_of_array(depth_mask, seg_mask1), majority_of_array(depth_mask, seg_mask2), majority_of_array(depth_mask, seg_mask3)
			sorted_items = sorted([(dep1, i), (dep2, j), (dep3, k)], key=lambda x: x[0])

			dep_diff1, dep_diff2 = abs(float(sorted_items[1][0]) - float(sorted_items[0][0])), abs(float(sorted_items[2][0]) - float(sorted_items[1][0]))

			if abs(dep_diff1 - dep_diff2) < self.threshold:
				continue

			if dep_diff1 > dep_diff2:
				if self.rng.choice([True, False]):
					anchor, closer, farther = sorted_items[-1][1], sorted_items[1][1], sorted_items[0][1]
				else:
					anchor, closer, farther = sorted_items[1][1], sorted_items[-1][1], sorted_items[0][1]
			else:
				if self.rng.choice([True, False]):
					anchor, closer, farther = sorted_items[0][1], sorted_items[1][1], sorted_items[-1][1]
				else:
					anchor, closer, farther = sorted_items[1][1], sorted_items[0][1], sorted_items[-1][1]

			anchor, closer, farther = labels[anchor], labels[closer], labels[farther]

			data_list.append((anchor, closer, farther))
		return data_list


class CloserObjectGenerator(CompareObjectDepthGenerator):
	"""
	"prompt"  : "Which of {candidates} is closer?",
	"response": "{closer}"
	"""

	qa_templates = get_qa_template("CloserObjectGenerator")
	des_templates = [
		{
			"description": "{closer} is closer than {farther}."
		}
	]
	metadata: bool = True

	def _generate(self, annotation: JointAnnotation) -> List:
		data_list = []
		for closer, farther in self.two_object(annotation):
			data_list += self.make_one_data(
				{
					"closer"    : closer,
					"farther"   : farther,
					"candidates": [closer, farther],
					"answer"    : closer,
					"metadata"  : {
						"object": [closer, farther]
					}
				},
			)
		return data_list


class FartherObjectGenerator(CompareObjectDepthGenerator):
	"""
	"prompt"  : "Which of {candidates} is farther?",
	"response": "{farther}"
	"""

	qa_templates = get_qa_template("FartherObjectGenerator")
	des_templates = [
		{
			"description": "{farther} is farther than {closer}."
		}
	]
	metadata: bool = True

	def _generate(self, annotation: JointAnnotation) -> List:
		data_list = []
		for closer, farther in self.two_object(annotation):
			data_list += self.make_one_data(
				{
					"farther"   : farther,
					"closer"    : closer,
					"candidates": [closer, farther],
					"answer"    : farther,
					"metadata"  : {
						"object": [closer, farther]
					}
				},
			)
		return data_list


class CloserToAnchorObjectGenerator(CompareObjectDepthGenerator):
	"""
	"prompt"  : "Which of {candidates} is closer to the {anchor}?",
	"response": "{closer}"
	"""

	qa_templates = get_qa_template("CloserToAnchorObjectGenerator")
	des_templates = [
		{
			"description": "{closer} is closer to {anchor} than {farther}."
		}
	]
	metadata: bool = True

	def _generate(self, annotation: JointAnnotation) -> List:
		data_list = []
		for anchor, closer, farther in self.three_object(annotation):
			data_list += self.make_one_data(
				{
					"anchor"    : anchor,
					"closer"    : closer,
					"farther"   : farther,
					"candidates": [closer, farther],
					"answer"    : closer,
					"metadata"  : {
						"object": [anchor, closer, farther]
					}
				},
			)
		return data_list


class FartherToAnchorObjectGenerator(CompareObjectDepthGenerator):
	"""
	"prompt"  : "Which of {candidates} is farther to the {anchor}?",
	"response": "{farther}"
	"""

	qa_templates = get_qa_template("FartherToAnchorObjectGenerator")
	des_templates = [
		{
			"description": "{farther} is farther to {anchor} than {closer}."
		}
	]
	metadata: bool = True

	def _generate(self, annotation: JointAnnotation) -> List:
		data_list = []
		for anchor, closer, farther in self.three_object(annotation):
			data_list += self.make_one_data(
				{
					"anchor"    : anchor,
					"closer"    : closer,
					"farther"   : farther,
					"candidates": [closer, farther],
					"answer"    : farther,
					"metadata"  : {
						"object": [anchor, closer, farther]
					}
				},
			)
		return data_list


ObjectDepthGenerator = [
	CloserObjectGenerator,
	FartherObjectGenerator,
	CloserToAnchorObjectGenerator,
	FartherToAnchorObjectGenerator
]
