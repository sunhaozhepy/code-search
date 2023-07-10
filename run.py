import random

import numpy as np
from torch.utils.data import Dataset

from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx

import os
import json
import ast

from tqdm import tqdm, trange

from transformers import AdamW, get_linear_schedule_with_warmup
from sentence_transformers import SentenceTransformer

import torch

from models import class_dict, viz_walker, GCN, GCN_Graph, Model


class TextDataset(Dataset):
  def __init__(self, data_path=None):
    self.code = []
    self.query = []
    with open(data_path, 'r') as f:
      for line in f:
        line = line.strip()
        js = json.loads(line)
        if 'code' in js:
          self.code.append(js['code'])
        else:
          self.code.append(js['function'])
        self.query.append(js['docstring'])

  def __len__(self):
    return len(self.code)

  def __getitem__(self, i):
    parsed = ast.parse(self.code[i])
    nw = viz_walker()
    nw.visit(parsed)
    code_data = from_networkx(nw.graph)
    return code_data, self.query[i]


def set_seed(seed=45):
  random.seed(seed)
  os.environ['PYHTONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True


def evaluate(model, dataloader):
    model.eval()
    code_vecs = [] 
    nl_vecs = []
    for batch in tqdm(dataloader):
        code_inputs = batch[0].to(device)    
        nl_inputs = batch[1]
        with torch.no_grad():
            code_vec, nl_vec = model(code_inputs, nl_inputs, return_vec=True)
            code_vecs.append(code_vec.cpu().numpy())
            nl_vecs.append(nl_vec.cpu().numpy())
    code_vecs = np.concatenate(code_vecs, 0)
    nl_vecs = np.concatenate(nl_vecs, 0)

    scores = np.matmul(nl_vecs, code_vecs.T)

    ranks = []
    for i in trange(len(scores)):
        score = scores[i,i]
        rank = 1
        for j in range(len(scores)):
            if i != j and scores[i, j] >= score:
                rank += 1
        ranks.append(1 / rank)    
    
    mrr = float(np.mean(ranks))

    return mrr


device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
print(f"using {device}.")

train_data_path = "./python/final/jsonl/train/python_train_cleaned.jsonl"
eval_data_path = "./python/final/jsonl/valid/python_valid_0_cleaned.jsonl"
output_dir = "./outputs"

# set_seed(45)

text_encoder = SentenceTransformer("all-mpnet-base-v2", device=device)
code_encoder = GCN_Graph(input_dim=len(class_dict), hidden_dim=768, num_layers=3, dropout=0.5)
model = Model(text_encoder, code_encoder)
model.to(device)

for param in text_encoder.parameters(): # we first test with fixed text encoder
  param.requires_grad = False

train_dataset = TextDataset(train_data_path)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=64)
eval_dataset = TextDataset(eval_data_path)
eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=64)

num_train_epochs = 10
save_steps = 1000

optimizer = AdamW(model.parameters(), lr=0.01)
scheduler = get_linear_schedule_with_warmup(optimizer, len(train_dataloader), len(train_dataloader) * num_train_epochs)

step = 0
best_mrr = 0.0

for idx in range(num_train_epochs):
  for batch in tqdm(train_dataloader):
    code_inputs = batch[0].to(device)
    nl_inputs = batch[1]

    model.train()
    loss, _, _ = model(code_inputs, nl_inputs)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    step += 1

    if step % save_steps == 0:
      print("Running evaluation...")
      mrr = evaluate(model, eval_dataloader)
      print(f"Mrr = {mrr}")
      if mrr >= best_mrr:
        best_mrr = mrr
        print(f"Save checkpoint!")

        checkpoint_prefix = 'checkpoint-best-mrr'
        output_dir = os.path.join(output_dir, checkpoint_prefix)
        if not os.path.exists(output_dir):
          os.makedirs(output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model

        torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))


device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
print(f"using {device}.")
 
test_data_file = "./python/final/jsonl/test/python_test_0_cleaned.jsonl"
output_dir = "./outputs"

# set_seed(45)

text_encoder = SentenceTransformer("all-mpnet-base-v2", device=device)
code_encoder = GCN_Graph(input_dim=len(class_dict), hidden_dim=768, num_layers=3, dropout=0.5)
model = Model(text_encoder, code_encoder)
model.load_state_dict(torch.load(os.path.join(output_dir, 'checkpoint-best-mrr/pytorch_model.bin')))
model.to(device)

test_dataset = TextDataset(test_data_file)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=64)

print("Running Test...")
mrr = evaluate(model, test_dataloader)
print(f"Mrr = {mrr}")