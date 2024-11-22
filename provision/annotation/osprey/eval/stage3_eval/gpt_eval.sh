NAME='osprey'
TYPE='description'

python osprey_generate_gpt_description_answer.py\
    --model ../../../exp/multi_region_v5_gqa_cot_bs16/ \
    --coco-img ../../../../images/coco/train2017 \
    --json ${TYPE}/questions.json

python eval_gpt.py\
    --question ${TYPE}/questions.json\
    --context ${TYPE}/prompt.json\
    --answer-list ${TYPE}/answers.json\
                  ${TYPE}/${NAME}_answer.json\
    --rule rule.json\
    --output result/gpt_score_${NAME}_${TYPE}.jsonl

python summarize_gpt_score.py --dir result

