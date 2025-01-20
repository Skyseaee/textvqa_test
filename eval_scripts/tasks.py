import abc
from dataclasses import asdict, dataclass
import os
from typing import Callable, Union

from load_dataset import load_storage_dataset, download_files


@dataclass
class TaskConfig():
    task: str = None
    task_alias: str = None
    tag: str = None

    # which dataset to use,
    # and what splits for what purpose
    dataset_path: str = None
    dataset_name: str = None
    dataset_kwargs: dict = None
    training_split: str = None
    validation_split: str = None
    test_split: str = None

    # scoring options
    metric_list: list = None
    output_type: str = "generate_until"
    generation_kwargs: dict = None
    repeats: int = 1
    filter_list: Union[str, list] = None

    def __post_init__(self) -> None:
            pass

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, item, value):
        return setattr(self, item, value)

    def to_dict(self):
        """dumps the current config as a dictionary object, as a printable format.
        null fields will not be printed.
        Used for dumping results alongside full task configuration

        :return: dict
            A printable dictionary version of the TaskConfig object.

        # TODO: should any default value in the TaskConfig not be printed?
        """
        cfg_dict = asdict(self)
        # remove values that are `None`
        for k, v in list(cfg_dict.items()):
            if v is None:
                cfg_dict.pop(k)
            elif isinstance(v, Callable):
                # TODO: this should handle Promptsource template objects as a separate case?
                cfg_dict[k] = str(v)
        return cfg_dict


class Task(abc.ABC):
    """A task represents an entire benchmark including its dataset, problems,
    answers, and evaluation methods. See BoolQ for a simple example implementation

    A `doc` can be any python object which represents one instance of evaluation.
    This is usually a dictionary e.g.
        {"question": ..., "answer": ...} or
        {"question": ..., question, answer)
    """
    
    VERSION = None

    DATASET_PATH: str = None
    DATASET_NAME: str = None

    OUTPUT_TYPE: str = None

    def __init__(self,
                 data_dir=None,
                 cache_dir=None,
                 download_mode=None,
                 from_hf=False,
                 config: TaskConfig=None) -> None:
        """
        :param data_dir: str
            Stores the path to a local folder containing the `Task`'s data files.
            Use this to specify the path to manually downloaded data (usually when
            the dataset is not publicly accessible).
        :param cache_dir: str
            The directory to read/write the `Task` dataset. This follows the
            HuggingFace `datasets` API with the default cache directory located at:
                `~/.datasets`
            NOTE: You can change the cache location globally for a given process
            to another directory:
                `export HF_DATASETS_CACHE="/path/to/another/directory"`
        :param download_mode: datasets.DownloadMode
            How to treat pre-existing `Task` downloads and data.
            - `datasets.DownloadMode.REUSE_DATASET_IF_EXISTS`
                Reuse download and reuse dataset.
            - `datasets.DownloadMode.REUSE_CACHE_IF_EXISTS`
                Reuse download with fresh dataset.
            - `datasets.DownloadMode.FORCE_REDOWNLOAD`
                Fresh download and fresh dataset.
        :param from_hf: bool
            Where to download the dataset
            - Default: False. dataset will be downloaded from ais dataset.
            - True: dataset will be downloaded from huggingface
        """
        self.from_hf = from_hf
        self.config = config
        # self.download(data_dir, cache_dir, download_mode, self.from_hf)
        
    
    def download(self, data_dir: str=None, download_config: dict=None, cache_dir: str='.dataset', download_mode=None, from_hf=False) -> str:
        if not from_hf:
            token = download_config.get("AIP_TOKEN", 'FcbgizlexopCBjrwsFuEjqhnfklyoFdr')

            dataset_id = download_config.get("datatset_id", '3746')
            dataset_commit = download_config.get("dataset_commit", 'a50f6734')
            dataset = load_storage_dataset(token, int(dataset_id), dataset_commit)
            return download_files(dataset, str(dataset_id) + dataset_commit, cache_dir=cache_dir)
        else:
            import datasets

            self.dataset = datasets.load_dataset(
                path=self.DATASET_PATH,
                name=self.DATASET_NAME,
                data_dir=data_dir,
                cache_dir=cache_dir,
                download_mode=download_mode,
            )

    @abc.abstractmethod
    def eval_task(self, *args, **kwargs):
        pass


class TaskRegistry:
    _registry = {}  # 存储任务名称与任务类的映射

    @classmethod
    def register(cls, task_name: str):
        def wrapper(task_cls):
            cls._registry[task_name] = task_cls
            return task_cls
        return wrapper

    @classmethod
    def get_task(cls, task_name: str, *args, **kwargs):
        if task_name not in cls._registry:
            raise ValueError(f"Task '{task_name}' is not registered.")
        return cls._registry[task_name](*args, **kwargs)

    @classmethod
    def list_tasks(cls):
        return list(cls._registry.keys())