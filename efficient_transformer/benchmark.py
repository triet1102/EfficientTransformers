from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from datasets import load_metric


class PerformanceBenchmark:
    def __init__(self, intents, pipeline, dataset, optim_type="BERT baseline"):
        self.intents = intents
        self.pipeline = pipeline
        self.dataset = dataset
        self.optim_type = optim_type

    def compute_size(self):
        state_dict = self.pipeline.model.state_dict()
        tmp_path = Path("model.pt")
        torch.save(state_dict, tmp_path)

        size_mb = Path(tmp_path).stat().st_size / (1024 * 1024)
        tmp_path.unlink()
        print(f"Model size (MB): {size_mb:.2f}")
        return {"size_mb": size_mb}

    def time_pipeline(self, query="What is the pin number for my account?"):
        latencies = []
        # warmup the cpu
        for _ in range(10):
            _ = self.pipeline(query)

        # timed up
        for _ in range(100):
            start_time = perf_counter()
            _ = self.pipeline(query)
            latency = perf_counter() - start_time
            latencies.append(latency)

        # compute run statistics
        time_avg_ms = 1000 * np.mean(latencies)
        time_std_ms = 1000 * np.std(latencies)

        print(f"Average lantency (ms) {time_avg_ms:.2f} +- {time_std_ms:.2f}")
        return {"time_avg_ms": time_avg_ms, "time_std_ms": time_std_ms}

    def compute_accuracy(self):
        accuracy_score = load_metric("accuracy")
        preds, labels = [], []
        for sample in self.dataset:
            pred = self.pipeline(sample["text"])[0]["label"]
            preds.append(self.intents.str2int(pred))
            labels.append(sample["intent"])

        accuracy = accuracy_score.compute(predictions=preds, references=labels)
        print(f"Accuracy on test set: {accuracy['accuracy']:.3f}")
        return accuracy

    def run_benchmark(self):
        metrics = {}
        metrics[self.optim_type] = self.compute_size()
        metrics[self.optim_type].update(self.time_pipeline())
        metrics[self.optim_type].update(self.compute_accuracy())

        return metrics
