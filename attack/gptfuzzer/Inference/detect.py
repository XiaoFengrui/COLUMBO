import os
import time
import copy
import numpy as np
import pandas as pd

from tqdm import tqdm
import argparse
import sys
sys.path.append('/home/ustc-5/XiaoF/AdvWebDefen/AdvSqli/mutate/')
# from zw_detector import ZWDetector
from ml_detector import MLDetector
import itertools

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3, 4'


parser = argparse.ArgumentParser()

parser.add_argument('--detector', '-d', required=True, default='CNN', choices=['CNN', 'LSTM', 'GenerAT', 'Zerowall', 'Wafbrain'], help='attack detector')
parser.add_argument('--dataset', '-ds', required=True, default='HPD', choices=['HPD', 'SIK'], help='choose dataset')
parser.add_argument("--training_ratio", "-tr", type=int, default=None, help="Training traio")
parser.add_argument("--ablation_selection", "-as",  default=None, choices=['RTDGAT', 'RTD', 'GAT'], help="Ablation Selection, without which modules")

input_args = parser.parse_args()

def read_payloads(path):
    linelen = 100
    # 读取前30000行
    with open(path, 'r', encoding='utf-8') as input_file:
        # 使用 itertools.islice 读取前30000行
        payloads = list(itertools.islice(input_file, linelen))
    return payloads

# def main():

#     np.random.seed(0)

#     # if input_args.detector in ['Zerowall', 'Wafbrain']:
#     #     # assert(not input_args.ML_url == 'default')
#     #     clsf = ZWDetector(input_args.detector, input_args.dataset)
#     if input_args.detector in ['CNN', 'cnn', 'GenerAT', 'generat', 'GENERAT', 'LSTM', 'lstm']:
#         clsf = MLDetector(input_args.detector, input_args.dataset)
        
#     # elif input_args.pattern == 'WAFaas':
#     #     assert(not input_args.WAFaas_url == 'default')
#     #     blackbox = True
#     #     clsf = WAFaasClient(base_url=input_args.WAFaas_url, sleep_time=0.001, cookie=input_args.cookie)
#     #     request_method = input_args.request_method

#     BENIGN_PAYLOADS_PATH = 'payload_benign.txt'
#     print("... load data from %s ..."%input_args.dataset)
#     payload_path = "/home/ustc-5/XiaoF/AdvWebDefen/gptfuzzer-main/Datasets/SQLi/SQLi_Dataset/SQLi_Dataset.txt" # "/home/ustc-5/XiaoF/AdvWebDefen/Dataset/" + input_args.dataset + "/test.tsv"
#     payloads = read_payloads(payload_path)
#     print("... success in loading %d samples from %s ..."%(len(payloads), input_args.dataset))
#     # payloads = [payloads[5]]
   
#     begin_time = time.time()
#     begin_time_str = time.strftime("%m-%d#%H-%M-%S", time.localtime())

#     counter = {'total': len(payloads), 'TP': 0}

#     TP = 0
#     for idx, payload in tqdm(enumerate(payloads)):
#         score = clsf.get_score(payload)
#         threshold = clsf.get_thresh()
#         if score >= threshold:
#             TP += 1
#     counter['TP'] = TP
#     end_time = time.time()
#     time_consume = end_time - begin_time

#     print("================Summary================")

#     # asr = counter['success'] / counter['total']
#     fnr = (counter['total'] - counter['TP'])/counter['total']
#     # print("Total payloads: {}, TP: {}, Failure: {}, Benign: {}, Except: {}, AttackSuccessRate: {}".format(
#     #     counter['total'], counter['success'], counter['failure'], counter['benign'], counter['except'], round(asr, 4)))
#     print("Total payloads: {}, TP: {}, FalseNegativeRate: {}".format(
#         counter['total'], counter['TP'], round(fnr, 6)))
#     print("Total time consuming: {}h/{}m/{}s".format(round(time_consume/3600, 4),
#                                                      round(time_consume/60, 4), round(time_consume, 4)))

def main():

    np.random.seed(0)

    # if input_args.detector in ['Zerowall', 'Wafbrain']:
    #     # assert(not input_args.ML_url == 'default')
    #     clsf = ZWDetector(input_args.detector, input_args.dataset, input_args.training_ratio)
    if input_args.detector in ['CNN', 'cnn', 'GenerAT', 'generat', 'GENERAT', 'LSTM', 'lstm']:
        clsf = MLDetector(input_args.detector, input_args.dataset, input_args.training_ratio, input_args.ablation_selection)
        
    # elif input_args.pattern == 'WAFaas':
    #     assert(not input_args.WAFaas_url == 'default')
    #     blackbox = True
    #     clsf = WAFaasClient(base_url=input_args.WAFaas_url, sleep_time=0.001, cookie=input_args.cookie)
    #     request_method = input_args.request_method

    print("... load data from %s ..."%input_args.dataset)
    payload_path = "../result/mutated_payload_" + input_args.dataset + ".csv"
    
    payload_df = pd.read_csv(payload_path, names=['idx', 'payload'], skiprows=1)
    # payload_df = payload_df.iloc[:100]
    payloads = []
    for idx, row in payload_df.iterrows():
        payloads.append([row['idx'], row['payload']])
    
    print("... success in loading %d samples from %s ..."%(len(payloads), input_args.dataset))
    # payloads = [payloads[5]]
   
    begin_time = time.time()
    begin_time_str = time.strftime("%m-%d#%H-%M-%S", time.localtime())

    counter = {'total': len(set(payload_df['idx'])), 'success': 0}
    flag = np.ones(len(set(payload_df['idx'])))
    # print(set(payload_df['idx']))

    TP = 0
    threshold = clsf.get_thresh()
    success_list = list(set(payload_df['idx']))
    for idx in tqdm(set(payload_df['idx'])):
        p_df = payload_df[payload_df['idx']==idx]
        p_df = p_df.sample(5)
        for _, rows in p_df.iterrows():
            score = clsf.get_score(rows['payload'])
            # print(idx, score, rows['payload'])
            if score >= threshold:
                flag[idx] = 0
                success_list.remove(idx)
                break
            
    counter['success'] = np.sum(flag==1)
    end_time = time.time()
    time_consume = end_time - begin_time

    print("================Summary================")

    asr = counter['success'] / counter['total']
    # fnr = (counter['total'] - counter['TP'])/counter['total']
    # print("Total payloads: {}, TP: {}, Failure: {}, Benign: {}, Except: {}, AttackSuccessRate: {}".format(
    #     counter['total'], counter['success'], counter['failure'], counter['benign'], counter['except'], round(asr, 4)))
    print("Total payloads: {}, success: {}, AttackSuccessRate: {}".format(
        counter['total'], counter['success'], round(asr, 6)))
    print("Total time consuming: {}h/{}m/{}s".format(round(time_consume/3600, 4),
                                                     round(time_consume/60, 4), round(time_consume, 4)))
    # print(success_list, len(success_list))


if __name__ == "__main__":
    main()


"""
python main.py -p WAFaas -WAFu http://10.15.196.135:8088/foo -pn multiple
python main.py -p WAFaas -WAFu http://10.15.196.135:8088/foo -pn multiple -g mcts
python main.py -p ML -MLu http://127.0.0.1:9001/wafbrain -MLt 0.1 -MLb blackbox_with_score -pn multiple -g mcts
python main.py -p ML -MLu http://127.0.0.1:9002/lstm -MLt 0.1 -MLb blackbox_with_score -pn multiple -g mcts
python main.py -p ML -MLu http://127.0.0.1:9003/cnn -MLt 0.1 -MLb blackbox_without_score -pn multiple -g random
"""