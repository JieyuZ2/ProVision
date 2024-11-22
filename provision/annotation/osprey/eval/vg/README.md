## Evaluation Code for Scene Graph Generation

Open-Vocabulary scene graph generation can be tricky due to the difficulty of acquiring classification score across the relation labels.
One would need to generate the relations from the MLLMs and run evaluation on the text generations, using tricks like semantic similarity with SentBert.
Alternatively, one can score the relation labels by getting the perplexity of language models, and sort by the reverse perplexity. 

### Scene Graph Prediction with SentBert Calibration

The following scripts generates scene graph prediction in text and runs SentBERT similarity score to acquire the calibration score.

Relationship Only for Given BBOX:
```
python -m osprey.eval.sg.eval_relation --model exp/relation_coco_sam_seem/ \
    --json data/sg/vg_test_sg_sam_hq_subset.jsonl \
    --temperature 0.5 \
    --top_p 0.9 \
    --output osprey/eval/results/relation/test/relation_coco_sam_seem-gt_objects_temp0.5_top0.9.pt # pytorch file
```

Scene Graph (with objects classes):
```
python -m osprey.eval.sg.eval_sg --model exp/vg_sg_v4_shuffle \
        --temperature 0.5 \
        --top_p 0.9 \
        --output osprey/eval/results/sg/test/vg_sg_v4_shuffle-latest-temp0.5_top0.9.pt # pytorch file
```

### Scene Graph Eval

Download the necessary files for evaluation: 
```
cd osprey/eval/sg/

wget https://storage.googleapis.com/ai2-jamesp-public/osprey/eval/eval_sg.zip

unzip eval_sg.zip
```

**Install maskrcnn**:
```
# APEX install
cd maskrcnn/apex
python setup.py build develop
cd ../..

# maskrcnn install
python setup.py clean --all
python setup.py build develop
```

Evaluate with Scene Graph benchmark:
```
python run_vg_evaluation.py --eval_results  osprey/eval/results/relation/test/relation_coco_sam_seem-gt_objects_temp0.5_top0.9.pt --mode sgcls
``` 

