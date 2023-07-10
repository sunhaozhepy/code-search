#! /bin/bash
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python.zip
unzip python.zip
gunzip ./python/final/jsonl/train/python_train_*.jsonl.gz
gunzip ./python/final/jsonl/valid/python_valid_*.jsonl.gz
gunzip ./python/final/jsonl/test/python_test_*.jsonl.gz

python ./preprocess.py