import numpy as np
import torch
from datasets import load_dataset, load_metric
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)


from efficient_transformer.trainer import (
    DistillationTrainer,
    DistillationTrainingArguments,
)

from efficient_transformer.benchmark import PerformanceBenchmark
from efficient_transformer.utility import (
    save_benchmark_result,
    load_benchmark_result,
    student_init,
    compute_metrics,
)
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

teacher_ckpt = "transformersbook/bert-base-uncased-finetuned-clinc"
pipe = pipeline("text-classification", model=teacher_ckpt)
print("OK")
student_ckpt = "distilbert-base-uncased"
student_tokenizer = AutoTokenizer.from_pretrained(student_ckpt)


def tokenize_text(batch):
    return student_tokenizer(batch["text"], truncation=True)


clinc = load_dataset("clinc_oos", "plus")
intents = clinc["test"].features["intent"]

clinc_enc = clinc.map(tokenize_text, batched=True, remove_columns=["text"])
clinc_enc = clinc_enc.rename_column("intent", "labels")

batch_size = 48
finetuned_ckpt = "distilbert-base-uncased-finetuned-clinc"
student_training_args = DistillationTrainingArguments(
    output_dir=finetuned_ckpt,
    evaluation_strategy="epoch",
    num_train_epochs=5,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    alpha=1,
    weight_decay=0.01,
    push_to_hub=True,
    logging_steps=len(clinc_enc["train"]) // batch_size,
    disable_tqdm=False,
    save_steps=1e9,
    log_level="warning",
)

id2label = pipe.model.config.id2label
label2id = pipe.model.config.label2id

num_labels = intents.num_classes
student_config = AutoConfig.from_pretrained(
    student_ckpt,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_model = AutoModelForSequenceClassification.from_pretrained(
    teacher_ckpt, num_labels=num_labels
).to(device)
del pipe

distilbert_trainer = DistillationTrainer(
    model_init=student_init,
    teacher_model=teacher_model,
    args=student_training_args,
    train_dataset=clinc_enc["train"],
    eval_dataset=clinc_enc["validation"],
    compute_metrics=compute_metrics,
    tokenizer=student_tokenizer,
)
print("Training....")
distilbert_trainer.train()
print("Finished Training!")
