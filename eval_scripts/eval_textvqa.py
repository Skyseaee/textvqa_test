from dataclasses import dataclass, field
import os
import json
import re
from typing import Dict, List

from eval_scripts.tasks import Task, TaskRegistry
from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator


@dataclass
class TextVQAConfig:
    task: str = "textvqa"
    annotation_file: str = "TextVQA_0.5.1_val.json"
    result_file: str = None
    dataset_path: str = "textvqa"
    dataset_name: str = "default"
    metric_list: List[str] = field(default_factory=lambda: ["accuracy"])


@TaskRegistry.register("textvqa")
class TextVQATask(Task):
    VERSION = 1.0
    DATASET_PATH = "textvqa"
    DATASET_NAME = "default"
    OUTPUT_TYPE = "generate_until"

    def __init__(self, config: TextVQAConfig = None, **kwargs):
        """
        Init TextVQA 
        :param config: TextVQAConfig 
        :param kwargs: additional kwargs, transport to Task
        """
        super().__init__(**kwargs)
        self.from_hf = False
        self.config = config if config else TextVQAConfig()
        # self.annotations = self._load_annotations(self.config.annotation_file)
        self.evaluator = TextVQAAccuracyEvaluator()

    def _load_annotations(self, annotation_file: str) -> Dict:
        """load annotations file"""
        with open(annotation_file, 'r') as f:
            data = json.load(f)['data']
        return {(item['image_id'], item['question'].lower()): item for item in data}

    @staticmethod
    def _prompt_processor(prompt: str) -> str:
        if prompt.startswith('OCR tokens: '):
            pattern = r"Question: (.*?) Short answer:"
            match = re.search(pattern, prompt, re.DOTALL)
            question = match.group(1)
        elif 'Reference OCR token: ' in prompt and len(prompt.split('\n')) == 3:
            if prompt.startswith('Reference OCR token:'):
                question = prompt.split('\n')[1]
            else:
                question = prompt.split('\n')[0]
        elif len(prompt.split('\n')) == 2:
            question = prompt.split('\n')[0]
        else:
            raise ValueError("Invalid prompt format")
        return question.lower()

    def eval_task(self, result_file: str = None, **kwargs):
        """
        :param result_file: path of result file
        :param kwargs: other kwargs
        """
        annotation_path = os.path.join(self.data_dir, os.path.join('s3upload', self.config.annotation_file))
        annotation_path = os.path.join(self.cache_path, annotation_path)
        self.annotations = self._load_annotations(annotation_path)

        result_file = result_file or self.config.result_file
        if not result_file:
            raise ValueError("Result file path is required for evaluation.")

        with open(result_file, 'r') as f:
            results = [json.loads(line) for line in f]

        pred_list = []
        for result in results:
            question_id = result['question_id']
            prompt = result['prompt']
            question = self._prompt_processor(prompt)
            annotation = self.annotations.get((question_id, question))
            if annotation:
                pred_list.append({
                    "pred_answer": result['text'],
                    "gt_answers": annotation['answers'],
                })

        # 计算准确率
        accuracy = self.evaluator.eval_pred_list(pred_list)
        print(f'Samples: {len(pred_list)}\nAccuracy: {100. * accuracy:.2f}%')
        return accuracy



# def prompt_processor(prompt):
#     if prompt.startswith('OCR tokens: '):
#         pattern = r"Question: (.*?) Short answer:"
#         match = re.search(pattern, prompt, re.DOTALL)
#         question = match.group(1)
#     elif 'Reference OCR token: ' in prompt and len(prompt.split('\n')) == 3:
#         if prompt.startswith('Reference OCR token:'):
#             question = prompt.split('\n')[1]
#         else:
#             question = prompt.split('\n')[0]
#     elif len(prompt.split('\n')) == 2:
#         question = prompt.split('\n')[0]
#     else:
#         assert False

#     return question.lower()


# def eval_single(annotation_file, result_file):
#     experiment_name = os.path.splitext(os.path.basename(result_file))[0]
#     print(experiment_name)
#     annotations = json.load(open(annotation_file))['data']
#     annotations = {(annotation['image_id'], annotation['question'].lower()): annotation for annotation in annotations}
#     results = [json.loads(line) for line in open(result_file)]

#     pred_list = []
#     for result in results:
#         annotation = annotations[(result['question_id'], prompt_processor(result['prompt']))]
#         pred_list.append({
#             "pred_answer": result['text'],
#             "gt_answers": annotation['answers'],
#         })

#     evaluator = TextVQAAccuracyEvaluator()
#     print('Samples: {}\nAccuracy: {:.2f}%\n'.format(len(pred_list), 100. * evaluator.eval_pred_list(pred_list)))


if __name__ == "__main__":
    # args = get_args()

    # if args.result_file is not None:
    #     eval_single(args.annotation_file, args.result_file)

    # if args.result_dir is not None:
    #     for result_file in sorted(os.listdir(args.result_dir)):
    #         if not result_file.endswith('.jsonl'):
    #             print(f'Skipping {result_file}')
    #             continue
    #         eval_single(args.annotation_file, os.path.join(args.result_dir, result_file))
    config = TextVQAConfig(
        annotation_file="path/to/annotations.json",
        result_file="path/to/results.json"
    )
    task = TextVQATask(config=config, data_dir="path/to/data", cache_dir="path/to/cache")
    task.eval_task()
