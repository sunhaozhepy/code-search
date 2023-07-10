import ast
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer

import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv, global_mean_pool

import numpy as np


# https://docs.python.org/3/library/ast.html#ast-helpers
# some of the classes may vary according to the version of Python, e.g. "Match" is not present before 3.10
class_list = [
  "AST",
  "Constant", "FormattedValue", "JoinedStr",
  "List", "Tuple", "Set", "Dict", "Name",
  "Load", "Store", "Del", "Starred", "Expr",
  "UnaryOp", "UAdd", "USub", "Not", "Invert",
  "BinOp", "Add", "Sub", "Mult", "Div", "FloorDiv",
  "Mod", "Pow", "LShift", "RShift", "BitOr",
  "BitXor", "BitAnd", "MatMult", "BoolOp",
  "And", "Or", "Compare", "Eq", "NotEq",
  "Lt", "LtE", "Gt", "GtE", "Is", "IsNot",
  "In", "NotIn", "Call", "keyword", "IfExp",
  "Attribute", "NamedExpr", "Subscript", "Slice",
  "ListComp", "SetComp", "GeneratorExp", "DictComp",
  "comprehension", "Assign", "AnnAssign", "AugAssign",
  "Raise", "Assert", "Delete", "Pass", "Import",
  "ImportFrom", "alias", "If", "For", "While",
  "Break", "Continue", "Try", "TryStar", "ExceptHandler",
  "With", "withitem", "Match", "match_case",
  "MatchValue", "MatchSingleton", "MatchSequence",
  "MatchStar", "MatchMapping", "MatchClass",
  "MatchAs", "MatchOr", "FunctionDef", "Lambda",
  "arguments", "arg", "Return", "Yield", "YieldFrom",
  "Global", "Nonlocal", "ClassDef", "AsyncFunctionDef",
  "Await", "AsyncFor", "AsyncWith"
]

class_dict = {class_list[i]: i for i in range(len(class_list))}


# based on https://gist.github.com/joshmarlow/4001898
# AST到NetworkX的转化，NetworkX到PyG图的转化，以及节点特征的构建
class viz_walker(ast.NodeVisitor):
  def __init__(self):
    self.stack = []
    self.graph = nx.Graph()
    self.vectorizer = CountVectorizer(lowercase=False, vocabulary=class_dict, binary=True)

  def generic_visit(self, stmt):
    node_name = str(stmt)

    parent_name = None

    if self.stack:
      parent_name = self.stack[-1]

    self.stack.append(node_name)

    self.graph.add_node(node_name, x=self.vectorizer.transform([stmt.__class__.__name__]).toarray().astype(np.float32).squeeze())

    if parent_name:
      self.graph.add_edge(node_name, parent_name)

    super(self.__class__, self).generic_visit(stmt)

    self.stack.pop()


# based on codebase of CS224W
class GCN(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, return_embeds=False):
    super(GCN, self).__init__()

    # A list of GCNConv layers
    self.convs = nn.ModuleList()

    # A list of 1D batch normalization layers
    self.bns = nn.ModuleList()

    # The log softmax layer
    self.softmax = nn.LogSoftmax(dim=-1)

    self.convs.append(GCNConv(input_dim, hidden_dim))
    for _ in range(num_layers - 2):
      self.convs.append(GCNConv(hidden_dim, hidden_dim))
    self.convs.append(GCNConv(hidden_dim, output_dim))

    for _ in range(num_layers - 1):
      self.bns.append(nn.BatchNorm1d(hidden_dim))

    self.relu = nn.ReLU()

    # Probability of an element getting zeroed
    self.dropout = nn.Dropout(p=dropout)

    # Skip classification layer and return node embeddings
    self.return_embeds = return_embeds

  def reset_parameters(self):
    for conv in self.convs:
      conv.reset_parameters()
    for bn in self.bns:
      bn.reset_parameters()

  def forward(self, x, adj_t):
    for i in range(len(self.convs) - 1):
      x = self.convs[i](x, adj_t) # 可以是edge list也可以是adjacency matrix
      x = self.bns[i](x)
      x = self.relu(x)
      x = self.dropout(x)

    out = self.convs[-1](x, adj_t) # 如果输出embedding，那么维度是output_dim
    if self.return_embeds == False:
      out = self.softmax(out)

    return out
  

### GCN to predict graph property
# based on codebase of CS224W
class GCN_Graph(nn.Module):
  def __init__(self, input_dim, hidden_dim, num_layers, dropout):
    super(GCN_Graph, self).__init__()

    # Node embedding model
    self.gnn_node = GCN(input_dim, hidden_dim,
      hidden_dim, num_layers, dropout, return_embeds=True)

    self.pool = global_mean_pool # 从节点embedding出发，用mean pooling得到graph embedding

  def reset_parameters(self):
    self.gnn_node.reset_parameters()
    self.linear.reset_parameters()

  def forward(self, batched_data):
    x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch

    out = self.gnn_node(x, edge_index)
    out = self.pool(out, batch)

    return out
  

class Model(nn.Module):  
  def __init__(self, text_encoder, code_encoder):
    super(Model, self).__init__()
    self.text_encoder = text_encoder
    self.code_encoder = code_encoder
    self.loss_fn = nn.CrossEntropyLoss()

    
  def forward(self, code_inputs, nl_inputs, return_vec=False): 
    nl_vec = self.text_encoder.encode(nl_inputs, convert_to_tensor=True)
    code_vec = self.code_encoder(code_inputs)
    if return_vec:
      return code_vec, nl_vec

    bs = nl_vec.shape[0]
    scores = (nl_vec[:, None, :] * code_vec[None, :, :]).sum(-1) # dot product of nl and code vectors as the score
    loss = self.loss_fn(scores, torch.arange(bs, device=scores.device))
    return loss, code_vec, nl_vec