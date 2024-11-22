#!/bin/bash

GPUS=2


MODEL_NAME=refexp_coco_fix
MODEL_NAME=relation_description_refexp_coco

MODEL_PATH=exp/${MODEL_NAME}

EVAL_PATH=osprey/eval/results/refexp/
OUTPUT=${EVAL_PATH}/${MODEL_NAME}-epoch1

output_file=${OUTPUT}.jsonl

GPU_OFFSET=2
for IDX in $(seq 0 $((GPUS-1))); do
    GPU_IDX=$((IDX + $GPU_OFFSET)) 
    echo $GPU_IDX
    CUDA_VISIBLE_DEVICES=$GPU_IDX python -m osprey.eval.refexp.eval_refexp \
        --model ${MODEL_PATH} \
        --output ${OUTPUT}_${IDX}.jsonl \
        --num_chunks $GPUS \
        --chunk_idx $IDX &
done

wait

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((GPUS-1))); do
    cat ${OUTPUT}_${IDX}.jsonl >> "$output_file"
done

python osprey/eval/refexp/run_refexp_eval.py \
    --result_file ${output_file}