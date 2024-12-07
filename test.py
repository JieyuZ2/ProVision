from provision.base import *
from provision.generators.multi_image.aggregate import *
from provision.generators.multi_image.compare import *
from provision.generators.multi_image.select import *


ANNOTATION_DIR = "notebooks/baseball_sample_annotations"
IMAGE_DIR = f"{ANNOTATION_DIR}/images"
N_WORKERS = 1


sample_dataset = JointDataset(
	[f"{ANNOTATION_DIR}/object-detection-annotations.json",
	 f"{ANNOTATION_DIR}/segmentation-annotations.json",
	 f"{ANNOTATION_DIR}/depth-estimation-annotations.json",
	 f"{ANNOTATION_DIR}/image-attributes-annotations.json",
	 f"{ANNOTATION_DIR}/relation-annotations-parsed.json"],
	seg_dir_path=f"{ANNOTATION_DIR}/seg_masks",
	depth_dir_path=f"{ANNOTATION_DIR}/depth_masks"
)

generator_list = (
	MultiSelectGeneratorList +
	MultiAggregateGeneratorList +
	MultiCompareGeneratorList
)

gen = JointGenerator(
	dataset=sample_dataset,
	generators=generator_list,
	template_mode='qa',
	return_templated=False,
	n_data=2,
	n_sample=1
)


qas = gen.generate(n_workers=N_WORKERS)

metadata_count = {}
for i in qas:
    for k, vs in i['metadata'].items():
        if k not in metadata_count:
            metadata_count[k] = {}
        for v in vs:
            if v not in metadata_count[k]:
                metadata_count[k][v] = 0
            metadata_count[k][v] += 1

print("Number of QAs: ", len(qas))

instructions = gen.multi_image_template(qas, multiple_choice_ratio=0.0)
mc_instructions = gen.multi_image_template(qas, multiple_choice_ratio=1.0)