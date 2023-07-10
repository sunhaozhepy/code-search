import os
import json
import ast

for root, dirs, files in os.walk("./python/final/jsonl"):
  for name in files:
    data = []
    file_name = os.path.join(root, name)
    with open(file_name, "r") as f:
      cleaned_name = os.path.splitext(file_name)[0] + "_cleaned" + os.path.splitext(file_name)[1]
      for line in f:
        line = line.strip()
        js = json.loads(line)
        try:
          if 'code' in js:
            code = js['code']
          else:
            code = js['function']
          parsed = ast.parse(code)
        except SyntaxError:
          pass
        else:
          data.append(line + '\n')     
    os.remove(file_name)
    last_line = data.pop(-1)
    data.append(last_line.strip('\n'))
    with open(cleaned_name, "w") as g:
      g.writelines(data)

total_data = ''
for file_name in os.listdir("./python/final/jsonl/train"):
  file_name = os.path.join("./python/final/jsonl/train", file_name)
  with open(file_name, 'r') as f:
    data = f.read()
    total_data += data
    total_data += '\n'
  os.remove(file_name)

with open("./python/final/jsonl/train/python_train_cleaned.jsonl", 'w') as g:
  g.write(total_data.strip('\n'))