# Scene Graph Evaluation

Our list of Scene Graph evaluators and evaluations are found in `osprey/eval/psg/scene_graph_evaluator.py`.
Command to run Model-based Scene Graph Eval:

```
python osprey/eval/psg/scene_graph_evaluator.py \
    --evaluator ${MODEL_NAME} \
    --data ${DATASET_NAME} \
    --output ${RESULTS_JSON}
```
If you want to add your own evaluator, make sure to register your class: `@SceneGraphEvaluator.register('your_eval')`
See [Model-based Evaluation](#model-based-evaluation) for more details.

### TODO:
- Add [Depth-Anythingv2 pipeline](https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf) to provide additional **depth** information about the image.

## Recall-based Evaluation

Prior works typically evaluate the quality scene graph based on on the **Recall@k** or **mean Recall(mR)@k**.
- Scene Graph Detection: Propose bounding box regions, and the correct relationship triplets (subject, predicate, object) are included. Bbox regions with **IoU > 0.5** are considered as candidates. 
- Scene Graph Classification: Given GT bounding box regions, identify the correct relationship triplet (subject, predicate, object).
- Predicate Classification: Given GT bounding box regions and object labels, identify the correct predicate.

You can refer to [SceneGraph-Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) for other list of metrics.

## [Model-based Evaluation](#model-based-evaluation)

Issue with **recall-based** metric is that the hallucinated relationships are not rightfully penalized, or sometimes the correct relationships are not rewarded due to the spare labels in the GT data. Ideally, we would like a metric that would capture the **precision** of the generated relationships, and one solution is to leverage on MLLMs such as GPT-4o to perform the evaluation.  However, it is not clear if these models are reliable experts capable of identifying correct and incorrect relationships. 

### Benchmark for Grounded Relationships
To support this, we curated a set of positive and negative relationship triplets: `[(subject, subject_bbox, predicate, object, object_bbox)]` that also require grounding skills, where models should check if the predicate holds true for subject and object in the specified locations.

You can refer to `data_generation/grounded_relation` to see how positive and negative are collected from GT data. For creating negatives, we either swap the relationships with the relationship that existed in the (subject and object) pairs. Or we shuffle the subject and object to enforce grounding capabilities.

**Data visualization**
```
python -m streamlit run data_generation/grounded_relation/visualize.py  --server.headless true --server.port 8501 --server.address 0.0.0.0
```

### GPT-4o

First, make sure you set Openai key in the env: `export OPENAI_API_KEY={YOUR_OPENAI_API_KEY}`

Evaluate GPT-4o on the grounded relationship benchmark.
By default, it uses bounding box coordinates and two images (original image and bbox drawn image).

```
python osprey/eval/psg/scene_graph_evaluator.py \
    --evaluator model_gpt4 \
    --data hico_det \
    --output osprey/eval/psg/results/hico_det_test_1000/gpt4o_results.json
```

**Supported Modes**
- `model_gpt4`: bbox coordinates & two images
- `model_gpt4_no_bbox`: no bbox coordinates & two imgaes
- `model_gpt4_no_region`: bbox coordinates & original image
- `model_gpt4_no_bbox_and_region`: no bbox coordinates & original image

### VLLMs

Open-sourced VLLMs can be also run locally to evaluate the grounded relationships. 

- Qwen2VL
```
python osprey/eval/psg/scene_graph_evaluator.py \
    --evaluator model_qwen2vl-7b \
    --data openimages \
    --output osprey/eval/psg/results/openimages_test_v6/qwen2vl-7b_results.json
```
