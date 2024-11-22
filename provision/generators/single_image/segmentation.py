from typing import List

import numpy as np

from ..template import get_qa_template
from ..utils import point_coordinate_to_ratio
from ...base import BaseGenerator
from ...dataset import SegmentationMask


def check_two_point(point1, point2, height, width):
	return point_coordinate_to_ratio(point1, height, width) == point_coordinate_to_ratio(point2, height, width)


def sample_points_with_mask(mask: np.ndarray, rng: np.random.RandomState) -> tuple:
	"""
	Sample 3 points from the mask
	same_point and diff_point has the same distance to the anchor_point
	"""

	y_points, x_points = np.where(mask == 1)
	height, width = mask.shape

	idx = rng.integers(0, len(y_points))
	anchor_point = (int(y_points[idx]), int(x_points[idx]))
	actual_anchor_point = str(point_coordinate_to_ratio((anchor_point[1], anchor_point[0]), height, width))

	meta_retry = 10
	while meta_retry:
		meta_retry -= 1

		idx = rng.integers(0, len(y_points))
		same_point = (int(y_points[idx]), int(x_points[idx]))
		actual_same_point = str(point_coordinate_to_ratio((same_point[1], same_point[0]), height, width))
		if actual_same_point == actual_anchor_point:
			continue

		length = np.sqrt((anchor_point[0] - same_point[0]) ** 2 + (anchor_point[1] - same_point[1]) ** 2)
		retry = 10
		while retry:
			retry -= 1
			angle = np.pi * rng.uniform(0, 2)
			diff_x = int(np.cos(angle) * length)
			diff_y = int(np.sin(angle) * length)
			diff_point = (anchor_point[0] + diff_y, anchor_point[1] + diff_x)
			actual_diff_point = str(point_coordinate_to_ratio((diff_point[1], diff_point[0]), height, width))
			if actual_diff_point == actual_anchor_point or actual_diff_point == actual_same_point:
				continue
			if (0 <= diff_point[0] < mask.shape[0] and 0 <= diff_point[1] < mask.shape[1]
					and mask[diff_point[0], diff_point[1]] == 0):
				# exchange x and y
				return (actual_anchor_point, actual_same_point, actual_diff_point)

	return ()


class ThreePointSegGenerator(BaseGenerator):

	def sample(self, annotation: SegmentationMask) -> List:
		if not isinstance(annotation, SegmentationMask):
			annotation = annotation.segment

		data_list = []
		label_mask = annotation.label_mask
		height, width = label_mask.shape
		for i in np.unique(label_mask):
			if i == 0:
				continue

			points = sample_points_with_mask(label_mask == i, self.rng)
			if len(points):
				data_list.append(points)

		return data_list


class SameObjectSegGenerator(ThreePointSegGenerator):
	"""
	"prompt"  : "Which point of {candidates} is in the same object with {anchor_point}?",
	"response": "{same_point}"
	"""

	qa_templates = get_qa_template("SameObjectSegGenerator")
	des_templates = [
		{
			"description": "Among points {candidates}, {same_point} is in the same object with {anchor_point}."
		}
	]
	metadata: bool = False

	def _generate(self, annotation: SegmentationMask, **kwargs) -> List:
		data_list = []

		for anchor_point, same_point, diff_point in self.sample(annotation):
			data_list += self.make_one_data(
				{
					"candidates"  : [same_point, diff_point],
					"answer"      : same_point,
					"anchor_point": anchor_point,
					"same_point"  : same_point,
					"diff_point"  : diff_point,
					"metadata"    : {}
				},
			)

		return data_list


class DiffObjectSegGenerator(ThreePointSegGenerator):
	"""
	"prompt"  : "Which point of {candidates} is not in the same object with {anchor_point}?",
	"response": "{diff_point}"
	"""

	qa_templates = get_qa_template("DiffObjectSegGenerator")
	des_templates = [
		{
			"description": "Among points {candidates}, {diff_point} is in the different object with {anchor_point}."
		}
	]
	metadata: bool = False

	def _generate(self, annotation: SegmentationMask, **kwargs) -> List:
		data_list = []

		for anchor_point, same_point, diff_point in self.sample(annotation):
			data_list += self.make_one_data(
				{
					"candidates"  : [same_point, diff_point],
					"answer"      : diff_point,
					"anchor_point": anchor_point,
					"same_point"  : same_point,
					"diff_point"  : diff_point,
					"metadata"    : {}
				},
			)

		return data_list


SegmentationGeneratorList = [
	SameObjectSegGenerator,
	DiffObjectSegGenerator,
]
