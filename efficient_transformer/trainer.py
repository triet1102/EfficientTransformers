import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments


class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature


class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model

    def compute_loss(self, model, inputs, return_outputs=False):
        # student outputs
        outputs_stu = model(**inputs)
        # cross-entropy loss and logits
        loss_ce = outputs_stu.loss
        logits_stu = outputs_stu.logits

        # extract logits from teacher
        with torch.no_grad():
            outputs_tea = self.teacher_model(**inputs)
            logits_tea = outputs_tea.logits

        # soften probabilities and compute loss
        loss_fct = nn.KLDivLoss(reduction="batchmean")

        # ! https://discuss.pytorch.org/t/kl-loss-and-log-softmax/69136/2
        # explaination about log_softmax for the student and softmax for the teacher
        loss_kd = self.args.temperature**2 * loss_fct(
            F.log_softmax(logits_stu / self.args.temperature, dim=-1),
            F.softmax(logits_tea / self.args.temperature, dim=-1),
        )

        # return the weighted student loss
        loss = self.args.alpha * loss_ce + (1.0 - self.args.alpha) * loss_kd
        return (loss, outputs_stu) if return_outputs else loss
