import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
import os
import pandas as pd
tqdm.pandas()
from transformers import GPT2Tokenizer
from gpt2 import GPT2HeadWithValueModel, respond_to_batch 
import time
import argparse
import itertools
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4'

def tokenize_and_pad(text, tokenizer, txt_in_len, pad_token_id):
    tokens = tokenizer.encode(text, return_tensors="pt").to(device)[0]
    if tokens.shape[0] < txt_in_len:
        tokens = F.pad(tokens, (0, txt_in_len - tokens.shape[0]), value=pad_token_id)  # 使用pad_token_id进行填充
    return tokens[:txt_in_len]


parser = argparse.ArgumentParser(description="")
parser.add_argument('--lm_name')
parser.add_argument('--ref_lm_name')
parser.add_argument('--mutate_times')
parser.add_argument('--txt_in_len')
parser.add_argument('--txt_out_len')
parser.add_argument('--dataPath', default="/home/ustc-5/XiaoF/AdvWebDefen/gptfuzzer-main/Datasets/SQLi/SQLi_Dataset/SQLi_Dataset.txt")
parser.add_argument('--savePath')
args = parser.parse_args()

lm_name = args.lm_name
ref_lm_name = args.ref_lm_name
mutate_times = int(args.mutate_times)
txt_in_len=int(args.txt_in_len)
txt_out_len=int(args.txt_out_len)
savePath=args.savePath
tokenizer = gpt2_tokenizer = GPT2Tokenizer.from_pretrained('/home/ustc-5/XiaoF/AdvWebDefen/gptfuzzer-main/openai/gpt2')
gpt2_model = GPT2HeadWithValueModel.from_pretrained(lm_name)
gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained(ref_lm_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)
_ = gpt2_model.eval()
_ = gpt2_model.to(device)
_ = gpt2_model_ref.eval()
_ = gpt2_model_ref.to(device)

inputfilePath = args.dataPath
if inputfilePath.endswith('.txt'):
    linelen = 30000
    # 读取前30000行
    with open(inputfilePath, 'r', encoding='utf-8') as input_file:
        # 使用 itertools.islice 读取前30000行
        sample_lines = list(itertools.islice(input_file, linelen))
elif inputfilePath.endswith('.tsv'):
    df = pd.read_csv(inputfilePath, delimiter='\t', names=['text', 'label'], skiprows=1)
    df = df[df['label'] == 1]
    # df = df.iloc[:100]
    sample_lines = list(df['text'])
    
bs = 64
wafData=pd.DataFrame()
# wafData['content']=['0' for _ in range(30000)]
wafData['content']=[item for item in sample_lines]
pad_token_id = gpt2_tokenizer.encode('<|endoftext|>')[0]
# for x in wafData['content']:
#     encode = tokenize_and_pad(x, gpt2_tokenizer, txt_in_len, pad_token_id)
#     print(encode, len(encode), '\n')
# exit(0)
# print(wafData['content'])
# exit(0)
wafData['tokens']=wafData['content'].progress_apply(lambda x: tokenize_and_pad(x, gpt2_tokenizer, txt_in_len, pad_token_id).to(device))
wafData['query'] = wafData['tokens'].progress_apply(lambda x: gpt2_tokenizer.decode(x))
# print(wafData['query'])

responseList = []
responsedict = {i: [] for i in range(len(wafData))}

starttime = time.time()
for mt in range(mutate_times):
    print('... Mutating Times: ', mt, '...')
    for start_idx in trange(0, len(wafData), bs):
        end_idx = min(start_idx + bs, len(wafData))
        df_batch = wafData.iloc[start_idx:end_idx]
        # print(f"Processing batch from {start_idx} to {end_idx}")
        # print(df_batch.shape)
        torch.cuda.empty_cache()
        query_tensors = torch.stack(df_batch['tokens'].tolist())
        # print('query shape: ', query_tensors.shape)
        response_tensors = respond_to_batch(gpt2_model, gpt2_model_ref, query_tensors, txt_len=txt_out_len)
        # print('response shape: ', response_tensors.shape)
        for i in range(len(df_batch)):
            responsedict[start_idx + i].append(gpt2_tokenizer.decode(response_tensors[i, :]).split('!')[0])
        # responseList += [gpt2_tokenizer.decode(response_tensors[i, :]).split('!')[0] for i in range(len(df_batch))]

endtime = time.time()
print("... Processing Time Cost %6.4fs ..."%(endtime - starttime))

data = []
for key, value_list in responsedict.items():
    for value in value_list:
        data.append([key, value])
df = pd.DataFrame(data, columns=['key', 'value'])
# 保存为 CSV 文件
df.to_csv(savePath,index=False)
# df_results = pd.DataFrame()

# df_results['response']=responseList
# df_results['query']='0'
# df_results['data']=df_results['query']+df_results['response']
# df_results[['data']].to_csv(savePath,index=False)


