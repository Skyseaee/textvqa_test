import argparse
import logging
import pandas as pd
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from PIL import Image
import math
import time
from torch.utils.data import Dataset, DataLoader
# from llava.constants import (
#     IMAGE_TOKEN_INDEX,
#     DEFAULT_IMAGE_TOKEN,
#     DEFAULT_IM_START_TOKEN,
#     DEFAULT_IM_END_TOKEN,
# )
# from llava.conversation import conv_templates, SeparatorStyle
# from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
# from llava.mm_utils import (
#     tokenizer_image_token,
#     process_images,
#     get_model_name_from_path,
# )

import api_server
import load_dataset as datasets


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder):
        self.questions = questions
        self.image_folder = image_folder

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]

        return os.path.join(self.image_folder, image_file), qs

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(
    questions,
    image_folder,
    batch_size=1,
    num_workers=4,
):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    return data_loader


def eval_model(args, path: str):
    # Model
    disable_torch_init()

    question_path = os.path.join(path, args.question_file)
    if question_path.endswith('csv'):
        df = pd.read_csv(os.path.join(path, args.question_file))
        questions = [row.to_dict() for _, row in df.iterrows()]
        # questions = [json.dumps(q) for q in questions]
    elif question_path.endswith('jsonl'):
        questions = [
            json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")
        ]
    else:
        logging.error(f"the format of file {args.question_file} not supported yet. please convert it to csv or jsonl file.")
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    # data_loader = create_data_loader(questions, args.image_folder)
    image_folder = os.path.join(path, args.image_folder)
    # server_addr = "http://sg9.aip.mlp.shopee.io/aip-svc-52/aigc-service-llm-7b-fp16"
    server_addr = "http://0.0.0.0:2333"
    api_client = api_server.APIClient(server_addr)

    for line in tqdm(questions, total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]
        model_name = "api"
        image_url = os.path.join(image_folder, line["image"])
        # print(image_url)

        if os.path.exists(image_url):
            base64_image = api_server.encode_image(image_url)
            image_url = f"data:image/jpeg;base64,{base64_image}"
        prompts = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": cur_prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            }
        ]
        outputs = ""
        begin_time = time.time()
        top_p = 1.0
        top_k = 1
        temperature = 0
        for output in api_client.v1_chat_completions(
            prompt=prompts,
            stream=False,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        ):
            outputs = output["choices"][0]["message"]["content"]
        lantency = time.time() - begin_time
        # print(lantency, outputs)
        ans_id = shortuuid.uuid()
        ans_file.write(
            json.dumps(
                {
                    "question_id": idx,
                    "prompt": cur_prompt,
                    "text": outputs,
                    "answer_id": ans_id,
                    "model_id": model_name,
                    "metadata": {},
                }
            )
            + "\n"
        )
        # ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
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
    args = parser.parse_args()

    token = os.getenv("AIP_TOKEN", 'FcbgizlexopCBjrwsFuEjqhnfklyoFdr')
    dataset_id = os.getenv("AIP_DATASET_ID", '3746')
    dataset_commit = os.getenv("AIP_DATASET_COMMIT_ID", 'a50f6734')
    if token and dataset_id and dataset_commit:
        logging.info(f'loading {dataset_id}-{dataset_commit} from AIS Dataset')

    dataset = datasets.load_storage_dataset(token, int(dataset_id), dataset_commit)
    folder = datasets.download_files(dataset, str(dataset_id) + dataset_commit)
    eval_model(args, folder)
