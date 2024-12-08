import os
import pickle as pkl
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import chain, combinations
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np

from .base import BaseDataset
from .generators.attribute_category import ATTRIBUTE_LIST


def normalize_attributes(attributes):
	if attributes is None:
		return None
	new_attributes = []
	for attr in attributes:
		new_attr = set()
		for a in attr:
			for ai in a.split(' and '):
				if ai in ATTRIBUTE_LIST:
					if ai == 'grey':
						ai = 'gray'
					new_attr.add(ai)
		new_attributes.append(list(new_attr))
	return new_attributes


EXCLUDE_RELATIONS = ['of']


def normalize_relations(relations):
	if relations is None:
		return None
	return [r for r in relations if r[1] not in EXCLUDE_RELATIONS]


def boxOverlap(box1, box2, vertical=False, horizontal=False):
	# get corners
	tl1 = (box1[0], box1[1])
	br1 = (box1[2], box1[3])
	tl2 = (box2[0], box2[1])
	br2 = (box2[2], box2[3])

	if horizontal:
		return not (tl1[0] >= br2[0] or tl2[0] >= br1[0])

	if vertical:
		return not (tl1[1] >= br2[1] or tl2[1] >= br1[1])

	# separating axis theorem
	# left/right
	if tl1[0] >= br2[0] or tl2[0] >= br1[0]:
		return False

	# top/down
	if tl1[1] >= br2[1] or tl2[1] >= br1[1]:
		return False

	# overlap
	return True


def boxInclude(box1, box2):
	# get corners
	tl1 = (box1[0], box1[1])
	br1 = (box1[2], box1[3])
	tl2 = (box2[0], box2[1])
	br2 = (box2[2], box2[3])

	# separating axis theorem
	# left/right
	if tl1[0] <= tl2[0] and tl1[1] <= tl2[1] and br1[0] >= br2[0] and br1[1] >= br2[1]:
		return True

	return False


@dataclass
class ClassLabel:
	"""
	For: Image Classification
	"""
	label: str


class AnnotationList:
	def subset(self, indices):
		data = {}
		for key, value in self.__dict__.copy().items():
			if key in self.__class__.__init__.__annotations__:
				if key == 'relations' and value is not None:
					relations = []
					for head, relation, target in value:
						if head in indices and target in indices:
							relations.append((indices.index(head), relation, indices.index(target)))
					data[key] = relations
				else:
					if isinstance(value, list) or isinstance(value, np.ndarray):
						data[key] = [value[i] for i in indices]
					else:
						data[key] = value
		return self.__class__(**data)

	def small_bboxes(self, ratio=0.5, height=None, width=None):
		if height is None:
			height = max([box[3] for box in self.bboxes])
		if width is None:
			width = max([box[2] for box in self.bboxes])
		area = height * width
		indices = [i for i, box in enumerate(self.bboxes) if ((box[2] - box[0]) * (box[3] - box[1])) < area * ratio]
		return self.subset(indices)

	def non_including_bboxes(self):
		assert hasattr(self, 'bboxes'), "No bboxes."

		non_included = []
		for i, box1 in enumerate(self.bboxes):
			include = False
			for j, box2 in enumerate(self.bboxes):
				if i != j and boxInclude(box1, box2):
					include = True
					break
			if not include:
				non_included.append(i)

		return self.subset(non_included)

	def non_overlapping_bboxes(self, vertical=False, horizontal=False):
		assert hasattr(self, 'bboxes'), "No bboxes."

		non_included = []
		for i, box1 in enumerate(self.bboxes):
			include = False
			for j, box2 in enumerate(self.bboxes):
				if i != j and boxInclude(box1, box2):
					include = True
					break
			if not include:
				non_included.append(i)

		overlapped, non_overlapped = set(), []
		for i, box1 in enumerate(self.bboxes):
			if i in non_included and i not in overlapped:
				overlap = False
				for j, box2 in enumerate(self.bboxes):
					if j in non_included and i != j and boxOverlap(box1, box2, vertical=vertical, horizontal=horizontal):
						overlap = True
						overlapped.add(i)
						overlapped.add(j)
						break
				if not overlap:
					non_overlapped.append(i)

		return self.subset(non_overlapped)

	def n_non_overlapping_bboxes(self, n, vertical=False, horizontal=False, unique=False):
		assert hasattr(self, 'bboxes'), "No bboxes."
		N = len(self.bboxes)

		if unique:
			object_labels = self.labels
			object_candidate_ids = [i for i, label in enumerate(object_labels) if object_labels.count(label) == 1]
		else:
			object_candidate_ids = list(range(N))

		adj_matrix = np.zeros((N, N))
		for i in range(N):
			box1 = self.bboxes[i]
			for j in range(i + 1, N):
				box2 = self.bboxes[j]
				if boxOverlap(box1, box2, vertical=vertical, horizontal=horizontal):
					adj_matrix[i, j] = 1
					adj_matrix[j, i] = 1

		object_combinations = combinations(object_candidate_ids, n)
		non_overlapped_n_set = []
		for combination in object_combinations:
			combination = list(combination)
			adj = adj_matrix[combination][:, combination]
			if not np.any(adj):
				non_overlapped_n_set.append(combination)

		return non_overlapped_n_set

	def attributed_bboxes(self):
		assert hasattr(self, 'bboxes'), "No bboxes."
		assert hasattr(self, 'attributes'), "No attributes."

		attributed = [i for i, attr in enumerate(self.attributes) if len(attr) > 0]
		return self.subset(attributed)


@dataclass
class Attributes(AnnotationList):
	bboxes: List[List[int]]
	labels: List[str]
	attributes: List[List[str]]
	scores: Optional[List[float]]

	def __len__(self):
		return len(self.labels)


@dataclass
class Relations(AnnotationList):
	"""
	relations: [(0, relation string, 1)]
	0 and 1 are id for bboxes and labels
	"""
	relations: List[Tuple[int, str, int]]
	bboxes: List[List[int]]
	labels: List[str]

	def __len__(self):
		return len(self.labels)


@dataclass
class SceneGraph(AnnotationList):
	"""
	Annotation: Tuple[head, target, relationship]
	For: Scene Graph
	"""
	bboxes: List[List[int]]
	labels: List[str]
	attributes: List[List[str]]
	relations: List[Tuple[int, str, int]]
	det_scores: Optional[List[float]]

	def __len__(self):
		return len(self.labels)

	@property
	def graph(self):
		if not hasattr(self, 'graph_'):
			self.graph_ = self._create_graph()
		return self.graph_

	def _create_graph(self):
		scene_graph = nx.MultiDiGraph()
		for i, label in enumerate(self.labels):
			scene_graph.add_node(i, value=label, attributes=self.attributes[i])
		for head, relation, target in self.relations:
			scene_graph.add_edge(head, target, value=relation)
		return scene_graph

	def single_edge_scene_graph(self, rng):
		uv_to_relations = {}
		for head, relation, target in self.relations:
			if head < target:
				head_to_target = True
			else:
				head, target = target, head
				head_to_target = False
			if (head, target) not in uv_to_relations:
				uv_to_relations[(head, target)] = []
			uv_to_relations[(head, target)].append((relation, head_to_target))
		relations = []
		for (head, target), rels in uv_to_relations.items():
			if len(rels) == 1:
				selected_rel, head_to_target = rels[0]
			else:
				selected_rel, head_to_target = rng.choice(rels)
			if head_to_target:
				relations.append((head, selected_rel, target))
			else:
				relations.append((target, selected_rel, head))
		return SceneGraph(bboxes=self.bboxes, labels=self.labels, attributes=self.attributes, relations=relations, det_scores=self.det_scores)

	def decompose(self) -> List:
		"""
		Decompose scene graph to multiple disconnected subgraphs.
		"""
		subgraphs = []
		G = self.graph
		connected_nodes = nx.connected_components(G.to_undirected())
		for ids in sorted(connected_nodes, key=len):
			ids = list(ids)
			bboxes = [self.bboxes[i] for i in ids]
			labels = [self.labels[i] for i in ids]
			attributes = [self.attributes[i] for i in ids]
			relations = []
			for head, relation, target in self.relations:
				if head in ids and target in ids:
					relations.append((ids.index(head), relation, ids.index(target)))
			if self.det_scores is not None:
				det_scores = [self.det_scores[i] for i in ids]
			else:
				det_scores = None
			graph = SceneGraph(bboxes=bboxes, labels=labels, attributes=attributes, relations=relations, det_scores=det_scores)
			subgraphs.append(graph)
		return subgraphs

	def draw(self):
		import matplotlib.pyplot as plt

		def bezier_curve(P0, P1, P2, t):
			return (1 - t) ** 2 * P0 + 2 * (1 - t) * t * P1 + t ** 2 * P2

		G = self.graph

		# Get node and edge labels
		node_labels = nx.get_node_attributes(G, 'label')
		edge_labels = {(u, v, key): data['label'] for u, v, key, data in G.edges(keys=True, data=True)}

		# Draw the graph
		pos = nx.spring_layout(G, k=3)  # Increase the value of k to spread out nodes
		plt.figure()
		# plt.figure(figsize=(8, 6))
		nx.draw(G, pos, with_labels=True, labels=node_labels, node_size=700, node_color='lightblue', font_size=10, font_weight='bold')

		# Draw curved edges with labels
		for (u, v, key), label in edge_labels.items():
			rad = 0.3 * (key - 1)  # Adjust the radius for each edge to avoid overlap
			P0, P2 = np.array(pos[u]), np.array(pos[v])
			ctrl_point = (P0 + P2) / 2 + rad * np.array([P2[1] - P0[1], P0[0] - P2[0]])

			# Compute points on the Bezier curve
			curve = np.array([bezier_curve(P0, ctrl_point, P2, t) for t in np.linspace(0, 1, 100)])
			plt.plot(curve[:, 0], curve[:, 1], 'k-')

			# Calculate the midpoint of the Bezier curve for the label
			mid_point = bezier_curve(P0, ctrl_point, P2, 0.5)
			plt.text(mid_point[0], mid_point[1], label, fontsize=10, color='red')

		plt.show()


# @dataclass
# class SegmentationMask(AnnotationList):
# 	"""
# 	For: Image Segmentation
# 	"""
# 	mask_id: Optional[str] = None
# 	dir_path: Optional[str] = None
# 	init_mask: Optional[np.ndarray] = None
# 	_label_mask = None
#
# 	@property
# 	def mask(self) -> np.ndarray:  # num_of_bboxes * H * W
# 		if self.init_mask is not None:
# 			return np.array(self.init_mask)
# 		else:
# 			path = f"{self.dir_path}/{self.mask_id}"
# 			if os.path.exists(path):
# 				if ".pkl" in self.mask_id:
# 					compressed_mask = pkl.load(open(path, 'rb'))
# 					self.init_mask = np.array([c_masks.toarray() for c_masks in compressed_mask], dtype=bool)
# 				elif ".npy" in self.mask_id:
# 					self.init_mask = np.load(path)
# 				else:
# 					raise ValueError(f"Unsupported mask format {self.mask_id}")
# 			else:
# 				filename = self.mask_id.split('/')[-1].split('.')[0]
# 				if os.path.exists(f"{self.dir_path}/{filename}.pkl"):
# 					compressed_mask = pkl.load(open(f"{self.dir_path}/{filename}.pkl", 'rb'))
# 					self.init_mask = np.array([c_masks.toarray() for c_masks in compressed_mask], dtype=bool)
# 				elif os.path.exists(f"{self.dir_path}/{filename}.npy"):
# 					self.init_mask = np.load(f"{self.dir_path}/{filename}.npy")
# 				else:
# 					raise ValueError(f"Unexistent mask file {self.mask_id}")
#
# 			return self.init_mask
#
# 	@property
# 	def label_mask(self) -> np.ndarray:  # H * W
# 		if self._label_mask is not None:
# 			return self._label_mask
#
# 		mask = self.mask
#
# 		label_mask = np.zeros((mask[0].shape[0], mask[0].shape[1]), dtype=np.uint8)
# 		i = 1  # 0 is reserved for background
# 		step = 1
# 		for ann in mask:
# 			label_mask[ann] = i
# 			i += step
# 		self._label_mask = label_mask
# 		return self._label_mask


@dataclass
class SegmentationMask(AnnotationList):
	"""
	For: Image Segmentation
	"""
	mask_id: Optional[str] = None
	dir_path: Optional[str] = None

	@property
	def mask(self) -> np.ndarray:  # num_of_bboxes * H * W
		path = f"{self.dir_path}/{self.mask_id}"
		if os.path.exists(path):
			if ".pkl" in self.mask_id:
				compressed_mask = pkl.load(open(path, 'rb'))
				return np.array([c_masks.toarray() for c_masks in compressed_mask], dtype=bool)
			elif ".npy" in self.mask_id:
				return np.load(path)
			else:
				raise ValueError(f"Unsupported mask format {self.mask_id}")
		else:
			filename = self.mask_id.split('/')[-1].split('.')[0]
			if os.path.exists(f"{self.dir_path}/{filename}.pkl"):
				compressed_mask = pkl.load(open(f"{self.dir_path}/{filename}.pkl", 'rb'))
				return np.array([c_masks.toarray() for c_masks in compressed_mask], dtype=bool)
			elif os.path.exists(f"{self.dir_path}/{filename}.npy"):
				return np.load(f"{self.dir_path}/{filename}.npy")
			else:
				raise ValueError(f"Unexistent mask file {self.mask_id}")

	@property
	def label_mask(self) -> np.ndarray:  # H * W
		mask = self.mask

		label_mask = np.zeros((mask[0].shape[0], mask[0].shape[1]), dtype=np.uint8)
		i = 1  # 0 is reserved for background
		step = 1
		for ann in mask:
			label_mask[ann] = i
			i += step
		return label_mask

# @dataclass
# class DepthMask(AnnotationList):
# 	"""
# 	For: Depth Estimation
# 	"""
# 	mask_id: Optional[str] = None
# 	dir_path: Optional[str] = None
# 	init_mask: Optional[np.ndarray] = None
#
# 	@property
# 	def mask(self):
# 		if self.init_mask is not None:
# 			return np.array(self.init_mask)
# 		else:
# 			if ".npy" in self.mask_id:
# 				self.init_mask = np.load(f"{self.dir_path}/{self.mask_id}")
# 			else:
# 				raise ValueError(f"Unsupported mask format {self.mask_id}")
# 			return self.init_mask
#
# 	@property
# 	def image_size(self):
# 		return self.mask.shape


@dataclass
class DepthMask(AnnotationList):
	"""
	For: Depth Estimation
	"""
	mask_id: Optional[str] = None
	dir_path: Optional[str] = None

	@property
	def mask(self):
		if ".npy" in self.mask_id:
			return np.load(f"{self.dir_path}/{self.mask_id}")
		else:
			raise ValueError(f"Unsupported mask format {self.mask_id}")

	@property
	def image_size(self):
		return self.mask.shape


@dataclass
class BoundBoxes(AnnotationList):
	bboxes: List[List[int]]
	labels: List[List[str]]
	scores: Optional[List[float]]
	depth: Optional[DepthMask]
	segment: Optional[SegmentationMask]

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, item):
		return self.bboxes[item], self.labels[item], self.scores[item], self.depth


@dataclass
class Caption:
	caption: str
	bbox: Optional[BoundBoxes] = None


@dataclass
class JointAnnotation(SceneGraph):
	"""
	Property:
		bboxes: Optional[List[List[int]]]
		labels: Optional[List[str]]
		attributes: Optional[List[List[str]]]
		relations: Optional[List[Tuple[int, str, int]]]
		scores: Optional[List[float]]
		caption: Optional[str]
		depth: Optional[DepthMask]
		segment: Optional[SegmentationMask]
	"""
	caption: Optional[str]
	depth: Optional[DepthMask]
	segment: Optional[SegmentationMask]  # num_of_bboxes * H * W
	height: int
	width: int


class JointDataset(BaseDataset):

	def __init__(self, annotation_path, depth_dir_path: str = None, seg_dir_path: str = None):
		self.depth_dir_path = depth_dir_path
		self.seg_dir_path = seg_dir_path
		super().__init__(annotation_path)

	def _sample(self, rng, l, n):
		if len(l) > n:
			return list(rng.choice(l, n, replace=False))
		elif len(l) == n:
			return l
		else:
			return []

	@property
	def object_to_id(self):
		if not hasattr(self, "_object_to_id"):
			self._object_to_id = defaultdict(lambda: {'ids': [], 'count': []})
			for i, annotation in enumerate(self.annotations):
				for label, cnt in Counter(annotation.labels).items():
					self._object_to_id[label]['ids'].append(i)
					self._object_to_id[label]['count'].append(cnt)

		return self._object_to_id

	def object_with_n_data(self, n):
		if not hasattr(self, "_object_with_n_data"):
			self._object_with_n_data = {n: [obj for obj, v in self.object_to_id.items() if len(v['ids']) >= n]}
		elif n not in self._object_with_n_data:
			self._object_with_n_data[n] = [obj for obj, v in self.object_to_id.items() if len(v['ids']) >= n]
		return self._object_with_n_data[n]

	def sample_object(self, rng):
		return rng.choice(list(self.object_to_id.keys()))

	def sample_data_and_obj(self, rng, n):
		candidates = self.object_with_n_data(n)
		obj = rng.choice(candidates)
		ids = self.object_to_id[obj]['ids']
		return obj, self._sample(rng, ids, n)

	def sample_data_with_obj(self, rng, obj, n):
		ids = self.object_to_id[obj]['ids']
		return self._sample(rng, ids, n)

	def sample_data_without_obj(self, rng, obj, n):
		ids = [i for i in range(len(self)) if i not in self.object_to_id[obj]['ids']]
		return self._sample(rng, ids, n)

	def sample_data_and_obj_diff_cnt(self, rng, n):
		candidates = self.object_with_n_data(n)
		obj = rng.choice(candidates)
		ids = self.object_to_id[obj]['ids']
		cnt = self.object_to_id[obj]['count']
		cnt2ids = defaultdict(list)
		for i, c in zip(ids, cnt):
			cnt2ids[c].append(i)
		if len(cnt2ids) >= n:
			selected = list(rng.choice(list(cnt2ids.keys()), n, replace=False))
			return obj, [(rng.choice(cnt2ids[cnt]), cnt) for cnt in selected]
		else:
			return []

	@property
	def relation_to_id(self):
		if not hasattr(self, "_relation_to_id"):
			_relation_to_id = defaultdict(dict)
			for i, annotation in enumerate(self.annotations):
				for o1, rel, o2 in annotation.relations:
					o1, o2 = annotation.labels[o1], annotation.labels[o2]
					if rel not in _relation_to_id[(o1, o2)]:
						_relation_to_id[(o1, o2)][rel] = set()
					_relation_to_id[(o1, o2)][rel].add(i)
			self._relation_to_id = {k: {rel: list(v) for rel, v in v.items()} for k, v in _relation_to_id.items()}
		return self._relation_to_id

	def relation_with_n_data(self, n):
		if not hasattr(self, "_relation_with_n_data"):
			self._relation_with_n_data = {n: [(objs, rel) for objs, v in self.relation_to_id.items() for rel, ids in v.items() if len(ids) >= n]}
		elif n not in self._relation_with_n_data:
			self._relation_with_n_data[n] = [(objs, rel) for objs, v in self.relation_to_id.items() for rel, ids in v.items() if len(ids) >= n]
		return self._relation_with_n_data[n]

	def object_pair_with_n_data(self, n):
		if not hasattr(self, "_object_pair_with_n_data"):
			self._object_pair_with_n_data = {n: []}
			for objs in self.relation_to_id.keys():
				ids = list(set(chain(*self.relation_to_id[objs].values())))
				if len(ids) >= n:
					self._object_pair_with_n_data[n].append((objs, ids))

		elif n not in self._object_pair_with_n_data:
			_object_pair_with_n_data = []
			for objs in self.relation_to_id.keys():
				ids = list(set(chain(*self.relation_to_id[objs].values())))
				if len(ids) >= n:
					_object_pair_with_n_data.append((objs, ids))
			self._object_pair_with_n_data[n] = _object_pair_with_n_data
		return self._object_pair_with_n_data[n]

	def sample_data_and_object_pair(self, rng, n):
		candidates = self.object_pair_with_n_data(n)
		(obj1, obj2), ids = candidates[rng.choice(len(candidates))]
		return obj1, obj2, self._sample(rng, ids, n)

	def sample_data_and_rel(self, rng, n):
		candidates = self.relation_with_n_data(n)
		objs, rel = candidates[rng.choice(len(candidates))]
		ids = self._relation_to_id[objs][rel]
		return objs, rel, self._sample(rng, ids, n)

	def sample_data_without_rel(self, rng, objs, ex_rel, n):
		candidates = set()
		ex_data = self._relation_to_id[objs][ex_rel]
		for rel in self._relation_to_id[objs]:
			if rel != ex_rel:
				candidates.update([i for i in self._relation_to_id[objs][rel] if i not in ex_data])
		return self._sample(rng, list(candidates), n)

	def _load(self):
		self.annotations = [
			JointAnnotation(
				height=label['height'],
				width=label['width'],
				bboxes=label.get('bboxes', None),
				labels=label.get('labels', None),
				attributes=normalize_attributes(label.get('attributes', None)),
				relations=normalize_relations(label.get('relations', None)),
				det_scores=label.get("det_scores", None),
				depth=DepthMask(mask_id=label.get("depth_mask_id", None), dir_path=self.depth_dir_path),
				segment=SegmentationMask(mask_id=label.get("seg_mask_id", None), dir_path=self.seg_dir_path),
				caption=label.get('caption', None)
			)
			for label in self.raw_annotations.copy()
		]
