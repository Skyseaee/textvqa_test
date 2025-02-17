import pandas as pd
import time
import os
import logging
from torchvision.transforms import Compose, ColorJitter, ToTensor, Resize
from torch.utils.data import DataLoader

from datatoolchain import load_from_hub, load_from_local, storage
from datatoolchain.datasets import AIStudioDataset
from datatoolchain.storage import DatasetStorage

def load_ais_dataset(token: str, dataset_id: int, dataset_commit: str,
                     split: str="train", cache_dir:str="~/.datasets") -> AIStudioDataset:
    logging.debug(f"dataset_id:{dataset_id}, dataset_commit:{dataset_commit}")

    cache_dir = os.path.expanduser(cache_dir)

    if os.path.exists(cache_dir):
        return load_from_local(local_path=cache_dir)

    dataset = load_from_hub(
        dataset_id=dataset_id,
        commit_id=dataset_commit,
        cache_dir=cache_dir,
        split=split,
        use_auth_token=token
    )

    return dataset


def load_storage_dataset(token: str, dataset_id: int, dataset_commit: str) -> DatasetStorage:
    os.environ['AIS_TOKEN'] = token or "FcbgizlexopCBjrwsFuEjqhnfklyoFdr"
    sto = storage.DatasetStorage.create_io_storage(dataset_id=dataset_id, commit_id=dataset_commit)
    return sto

def download_files(sto: DatasetStorage, dataset_info: str, cache_dir='.dataset'):
    path = os.path.join(cache_dir, dataset_info)

    if os.path.exists(path):
        return path
    else:
        # print(path, filename)
        sto._cache.download(path)
        return path


if __name__ == '__main__':
    token = os.getenv("AIP_TOKEN", 'FcbgizlexopCBjrwsFuEjqhnfklyoFdr')
    dataset_id = os.getenv("AIP_DATASET_ID", '3746')
    dataset_commit = os.getenv("AIP_DATASET_COMMIT_ID", 'a50f6734')
    if token and dataset_id and dataset_commit:
        logging.info(f'loading {dataset_id}-{dataset_commit} from AIS Dataset')

    dataset = load_storage_dataset(token, int(dataset_id), dataset_commit)
    download_files(dataset, str(dataset_id) + dataset_commit)
    # dataset.map()
