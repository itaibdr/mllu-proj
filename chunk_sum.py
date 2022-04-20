import pandas as pd
!pip install transformers
!pip install sentencepiece
from tqdm import tqdm
import torch

#!wget https://raw.githubusercontent.com/nyu-mll/quality/main/data/QuALITY.v0.9.htmlstripped.train

data = pd.read_json('/content/QuALITY.v0.9.htmlstripped.train', lines = True)

articles = [i.replace('\n','').replace('     ','') for i in data['article']]
chunks = []
for j in range(len(articles)):
  chunks.append([articles[j][i:i+2000] for i in range(0, len(articles[j]), 2000)])

device = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import BartTokenizer, BartForConditionalGeneration

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
model = model.to(device)

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

all_inputs = []
for j in tqdm(range(len(chunks))):
  inputs = []
  for i in range(len(chunks[j])):
    inputs.append(tokenizer(chunks[j][i], return_tensors="pt").to(device))
  all_inputs.append(inputs)


all_sums = []
for j in tqdm(range(len(all_inputs[0:10]))):
  sums = []
  ids = []
  for i in range(len(all_inputs[j])):
    inputs = all_inputs[j]
    if i == 0:
      summary_ids = model.generate(inputs[i]["input_ids"], num_beams=2)
      ids.append(summary_ids)
      sums.append(tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0])
      print(ids[i].shape)
    else:
      last_sum = tokenizer.batch_decode(ids[i-1], skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
      last_sum = last_sum.split('. ')
      apt = tokenizer(last_sum[-1],return_tensors ='pt').to(device)
      summary_ids = model.generate(torch.concat((apt['input_ids'],inputs[i]["input_ids"]),1), num_beams=2, max_length=128)
      ids.append(summary_ids)
      sums.append(tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0])
  all_sums.append(' '.join(sums))
