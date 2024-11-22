from typing import List

from ..template import get_qa_template
from ..utils import point_coordinate_to_ratio
from ...base import BaseGenerator
from ...dataset import DepthMask


class ComparePointDepthGenerator(BaseGenerator):
	def __init__(self, dataset, threshold=50, **kwargs):
		super().__init__(dataset=dataset, **kwargs)
		self.threshold = threshold

	def _sample_points(self, annotation: DepthMask):

		if not isinstance(annotation, DepthMask):
			annotation = annotation.depth
		depth_mask = annotation.mask
		height, width = depth_mask.shape

		point1 = self.rng.integers(0, height), self.rng.integers(0, width)
		point2 = self.rng.integers(0, height), self.rng.integers(0, width)
		actual_point1 = str(point_coordinate_to_ratio((point1[1], point1[0]), height, width))
		actual_point2 = str(point_coordinate_to_ratio((point2[1], point2[0]), height, width))

		while actual_point1 == actual_point2:
			point2 = self.rng.integers(0, height), self.rng.integers(0, width)
			actual_point1 = str(point_coordinate_to_ratio((point1[1], point1[0]), height, width))
			actual_point2 = str(point_coordinate_to_ratio((point2[1], point2[0]), height, width))

		dep1, dep2 = float(depth_mask[point1]), float(depth_mask[point2])
		retry = 10
		while abs(dep1 - dep2) < self.threshold and retry > 0:
			point1 = self.rng.integers(0, height), self.rng.integers(0, width)
			point2 = self.rng.integers(0, height), self.rng.integers(0, width)
			dep1, dep2 = float(depth_mask[point1]), float(depth_mask[point2])
			retry -= 1

		if depth_mask[point1] < depth_mask[point2]:
			closer, farther = actual_point1, actual_point2
		else:
			closer, farther = actual_point2, actual_point1

		return closer, farther


class CloserPointGenerator(ComparePointDepthGenerator):
	"""
	"prompt"  : "Which point of {candidates} is closer?",
	"response": "{closer}"
	"""

	qa_templates = get_qa_template("CloserPointGenerator")
	des_templates = [
		{
			"description": "Point {closer} is closer than {farther}."
		}
	]
	metadata: bool = False

	def _generate(self, annotation: DepthMask, **kwargs) -> List:
		closer, farther = self._sample_points(annotation)

		return self.make_one_data(
			{
				"candidates": [closer, farther],
				"answer"    : closer,
				"closer"    : closer,
				"farther"   : farther,
				"metadata"  : {}
			},
		)


class FartherPointGenerator(ComparePointDepthGenerator):
	"""
	"prompt"  : "Which point of {candidates} is closer?",
	"response": "{farther}"
	"""

	qa_templates = get_qa_template("FartherPointGenerator")
	des_templates = [
		{
			"description": "Point {farther} is further than {closer}."
		}
	]
	metadata: bool = False

	def _generate(self, annotation: DepthMask, **kwargs) -> List:
		closer, farther = self._sample_points(annotation)

		return self.make_one_data(
			{
				"candidates": [closer, farther],
				"answer"    : farther,
				"closer"    : closer,
				"farther"   : farther,
				"metadata"  : {}
			},
		)


DepthGeneratorList = [
	CloserPointGenerator,
	FartherPointGenerator,
]
