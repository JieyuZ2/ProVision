from typing import List, Dict
import argparse
import pandas as pd
import numpy as np
import json
import os
from osprey.eval.eval import OspreyEval

class Metric:

    def __init__(self):
        pass

def get_exact_acc(df) -> float:
    acc = (df['pred_answer'] == df['answer']).mean()
    print('Accuracy:', acc)

    # Per Category
    if 'category' in df:
        acc_per_category = df.groupby('category').apply(lambda x: (x['pred_answer'] == x['answer']).mean())
        print('Accuracy per category:')
        print(acc_per_category)
    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result_file", type=str)
    parser.add_argument("--metric", type=str, choices=['exact_acc'], default='exact_acc')
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--task_name", type=str, default=None)

    # wandb logs
    parser.add_argument('--log_wandb', action='store_true', help='Log results to wandb')
    parser.add_argument("--wandb_run_name", type=str)
    parser.add_argument("--wandb_key", type=str)
    args = parser.parse_args()

    df = pd.read_json(args.result_file, lines=True)
    print('Task:', args.task_name)
    acc = get_exact_acc(df)

    # save to output
    if args.output_file:
        assert args.task_name is not None, "Task name is required to save output"
        output = {}
        # update existing output file with new results
        if os.path.isfile(args.output_file):
            with open(args.output_file, 'r') as f:
                output = json.load(f)
        result = {args.task_name: acc}
        output.update(result)
        print('Saving result to.. {}'.format(args.output_file))
        print(output)
        with open(args.output_file, 'w') as f:
            json.dump(output, f)
    
    # result: List[Dict] = .to_dict(orient='records')
    # if args.log_wandb:
    #     OspreyEval.log_wandb(args.wandb_run_name, args.wandb_key, {args.wandb_key: acc})
