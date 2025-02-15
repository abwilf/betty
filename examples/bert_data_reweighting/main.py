import os
import argparse
import pandas as pd

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup
from transformers.optimization import AdamW

from model import BertModel, MLP
from utils import DataPrecessForSentence, correct_predictions, set_seed, split_dataset

from betty.engine import Engine
from betty.problems import ImplicitProblem
from betty.configs import Config, EngineConfig


parser = argparse.ArgumentParser(description="Meta_Weight_Net")
parser.add_argument("--baseline", action="store_true")
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--distributed", action="store_true")
parser.add_argument("--rollback", action="store_true")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--meta_net_hidden_size", type=int, default=500)
parser.add_argument("--meta_net_num_layers", type=int, default=1)

parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--patience", type=int, default=0)
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--meta_lr", type=float, default=1e-4)
parser.add_argument("--meta_weight_decay", type=float, default=1e-4)

parser.add_argument("--imbalance_factor", type=int, default=10)
parser.add_argument("--max_seq_len", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=32)

args = parser.parse_args()
print(args)

device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(args.seed)

data_path = "./data/"
train_df = pd.read_csv(
    os.path.join(data_path, "train.tsv"),
    sep="\t",
    header=None,
    names=["similarity", "s1"],
)
dev_df = pd.read_csv(
    os.path.join(data_path, "dev.tsv"),
    sep="\t",
    header=None,
    names=["similarity", "s1"],
)
test_df = pd.read_csv(
    os.path.join(data_path, "test.tsv"),
    sep="\t",
    header=None,
    names=["similarity", "s1"],
)

# Finetune
bertmodel = BertModel(requires_grad=True)
tokenizer = bertmodel.tokenizer
train_data = DataPrecessForSentence(tokenizer, train_df, max_seq_len=args.max_seq_len)
train_data, meta_data = split_dataset(
    train_data, imbalance_factor=args.imbalance_factor
)
train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
epoch_len = len(train_loader)
param_optimizer = list(bertmodel.named_parameters())
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [
            p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.01,
    },
    {
        "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epoch_len, gamma=0.85)

# Reweight
meta_net = MLP(
    in_size=2,
    hidden_size=args.meta_net_hidden_size,
    num_layers=args.meta_net_num_layers,
)
meta_loader = DataLoader(meta_data, shuffle=True, batch_size=args.batch_size)
meta_optimizer = torch.optim.Adam(
    meta_net.parameters(),
    lr=args.meta_lr,
    weight_decay=args.meta_weight_decay,
)

# valid
dev_data = DataPrecessForSentence(tokenizer, dev_df, max_seq_len=args.max_seq_len)
dev_loader = DataLoader(dev_data, shuffle=True, batch_size=args.batch_size)

# test
test_data = DataPrecessForSentence(tokenizer, test_df, max_seq_len=args.max_seq_len)
test_loader = DataLoader(test_data, shuffle=False, batch_size=args.batch_size)


class Finetune(ImplicitProblem):
    def training_step(self, batch):
        seqs, masks, segments, labels = batch
        _, logits, probs = self.module(seqs, masks, segments, labels)
        loss_vector = F.cross_entropy(logits.view(-1, 2), labels, reduction="none")
        if args.baseline:
            loss = torch.mean(loss_vector)
        else:
            loss_vector_reshape = torch.reshape(loss_vector, (-1, 1))
            weight = self.reweight(probs.detach())
            loss = torch.mean(weight * loss_vector_reshape)
        return loss


class Reweight(ImplicitProblem):
    def training_step(self, batch):
        seqs, masks, segments, labels = batch
        loss, *_ = self.finetune(seqs, masks, segments, labels)
        return loss


best_acc = -1


class SSTEngine(Engine):
    @torch.no_grad()
    def validation(self):
        running_loss = 0.0
        running_accuracy = 0.0
        all_prob = []
        all_labels = []
        global best_acc
        for (
            batch_seqs,
            batch_seq_masks,
            batch_seq_segments,
            batch_labels,
        ) in dev_loader:
            seqs = batch_seqs.to(device)
            masks = batch_seq_masks.to(device)
            segments = batch_seq_segments.to(device)
            labels = batch_labels.to(device)
            loss, logits, probabilities = self.finetune(seqs, masks, segments, labels)
            running_loss += loss.item()
            running_accuracy += correct_predictions(probabilities, labels)
            all_prob.extend(probabilities[:, 1].cpu().numpy())
            all_labels.extend(batch_labels)

        valid_loss = running_loss / len(dev_loader)
        valid_accuracy = running_accuracy / (len(dev_loader.dataset))
        if best_acc < valid_accuracy:
            best_acc = valid_accuracy
        return {"loss": valid_loss, "acc": valid_accuracy, "best_acc": best_acc}


engine_config = EngineConfig(train_iters=200, valid_step=50)
finetune_config = Config(
    type="darts", fp16=args.fp16, retain_graph=True, gradient_clipping=10.0
)
reweight_config = Config(
    type="darts", fp16=args.fp16, unroll_steps=1, gradient_clipping=10.0
)

finetune = Finetune(
    name="finetune",
    module=bertmodel,
    optimizer=optimizer,
    # scheduler=scheduler,
    train_data_loader=train_loader,
    config=finetune_config,
)
reweight = Reweight(
    name="reweight",
    module=meta_net,
    optimizer=meta_optimizer,
    train_data_loader=meta_loader,
    config=reweight_config,
)

if args.baseline:
    problems = [finetune]
    u2l, l2u = {}, {}
else:
    problems = [reweight, finetune]
    u2l = {reweight: [finetune]}
    l2u = {finetune: [reweight]}
dependencies = {"l2u": l2u, "u2l": u2l}


engine = SSTEngine(config=engine_config, problems=problems, dependencies=dependencies)
engine.run()
