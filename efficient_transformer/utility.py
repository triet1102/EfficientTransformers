import pickle
import os
from transformers import AutoModelForSequenceClassification
from transformers import PretrainedConfig
from datasets import load_metric
import numpy as np

def save_benchmark_result(res: dict, file_path: str) -> None:
    if os.path.exists(file_path):
        os.remove(file_path)
    else:
        with open(file_path, 'wb') as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_benchmark_result(file_path: str) -> dict:
    with open(file_path, 'rb') as handle:
        res = pickle.load(handle)
    
    return res
    
