import argparse
import logging
import os

from eval_benchmarks import inference_model
from eval_scripts.tasks import Task, TaskRegistry
from eval_scripts.eval_textvqa import TextVQATask


def parser_task_name(tasks: str):
    return tasks.lower().split(',')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--task-name", type=str, default="textvqa")
    parser.add_argument("--image-folder", type=str, default="s3upload/images")
    parser.add_argument("--question-file", type=str, default="s3upload/valid.csv")
    parser.add_argument("--answers-file", type=str, default="./answer_file/answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--api-address", type=str, default="http://localhost:23333")

    parser.add_argument('--annotation-file', type=str)
    parser.add_argument('--result-file', type=str, default="../answer_file/answer.jsonl")
    parser.add_argument('--result-dir', type=str)
    return parser.parse_args()


def main():
    args = get_args()

    tasks = parser_task_name(args.task_name)

    for task_name in tasks:
        task: Task = TaskRegistry.get_task(task_name)
        download_config = dict()
        folder = ""
        if not task.from_hf:
            dataset_id = os.getenv("AIP_DATASET_ID", '3746')
            dataset_commit = os.getenv("AIP_DATASET_COMMIT_ID", 'a50f6734')
            download_config["datatset_id"] = dataset_id
            download_config["datatset_commit"] = dataset_commit
            folder = task.download(download_config=download_config)
        else:
            # download dataset from hf
            pass

        result_file = inference_model(args, folder)
        task.eval_task(result_file)


if __name__ == '__main__':
    main()