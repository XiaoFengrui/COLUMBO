import click
import pickle
import re
# from wafamole.evasion import EvasionEngine
# from wafamole.evasion.random import RandomEvasionEngine
from evasion.eva_mutation import EvasionEngineNew, RandomEvasionEngineNew
# from exceptions.models_exceptions import UnknownModelError
# from wafamole.models import TokenClassifierWrapper, WafBrainWrapper, SQLiGoTWrapper, MLBasedWAFWrapper
# try:
#     from wafamole.models.modsec_wrapper import PyModSecurityWrapper
# except ImportError:
#     # ModSecurity module is not available
#     pass
import argparse
import sys
import numpy as np
import pandas as pd
sys.path.append('/home/ustc-5/XiaoF/AdvWebDefen/AdvSqli/mutate/')
# from zw_detector import ZWDetector
from ml_detector import MLDetector
# import multiprocessing as mp
from tqdm import tqdm
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = '4'

def evade(clsf, payload, max_rounds, round_sizes, timeout, random_engine, output_path):

    threshold = clsf.get_thresh()
    engine = RandomEvasionEngineNew(clsf) if random_engine is not None else EvasionEngineNew(clsf)
    query_body = payload
    if random_engine is not None:
        random_results = []
        for i in range(int(random_engine)):
            engine.evaluate(query_body, max_rounds, 1, timeout, threshold)
            random_results.append(engine.transformations)
            print("Round {} done".format(i))
        if output_path is not None:
            with open(output_path, 'wb') as out_file:
                pickle.dump(random_results, out_file)
    else:
       return engine.evaluate(query_body, max_rounds, round_sizes, timeout, threshold)
        
def read_payloads_csv(path):
    # 读取TSV文件
    df = pd.read_csv(path, delimiter='\t', names=['payload', 'label'], skiprows=1)
    
    # 过滤label为1的行
    filtered_df = df[df['label'] == 1]
    
    # 提取payload列并加上空格
    payloads = (filtered_df['payload'] + ' ').tolist()
    labels = filtered_df['label'].astype(int).tolist()
    
    return payloads, labels
        
def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector', '-d', required=True, default='CNN', choices=['CNN', 'LSTM', 'GenerAT', 'Zerowall', 'Wafbrain'], help='attack detector')
    parser.add_argument('--dataset', '-ds', required=True, default='HPD', choices=['HPD', 'SIK'], help='choose dataset')
    parser.add_argument("--timeout", "-t", default=240, type=int, help="Timeout when evading the model")
    parser.add_argument("--max_rounds", "-r", default=5, type=int, help="Maximum number of fuzzing rounds. Default: 1000") # steps
    parser.add_argument("--round_size", "-s", default=5, type=int, help="Fuzzing step size for each round (parallel fuzzing steps). Default: 20") # budget
    parser.add_argument("--random_engine", default=None, help="Use random transformations instead of evolution engine. Set the number of trials")
    parser.add_argument("--output_path", default=None, help="Location were to save the results of the random engine. NOT USED WITH REGULAR EVOLUTION ENGINE")
    parser.add_argument("--training_ratio", "-tr", type=int, default=None, help="Training traio")
    parser.add_argument("--ablation_selection", "-as",  default=None, choices=['RTDGAT', 'RTD', 'GAT'], help="Ablation Selection, without which modules")
    
    np.random.seed(0)
    input_args = parser.parse_args()
    if input_args.detector in ['CNN', 'cnn', 'GenerAT', 'generat', 'GENERAT', 'LSTM', 'lstm']:
        clsf = MLDetector(input_args.detector, input_args.dataset, tr = input_args.training_ratio, ablation = input_args.ablation_selection)
    # if input_args.detector in ['Zerowall', 'Wafbrain']:
    #     clsf = ZWDetector(input_args.detector, input_args.dataset)
    
    print("... load data from %s ..."%input_args.dataset)
    payload_path = "/home/ustc-5/XiaoF/AdvWebDefen/Dataset/" + input_args.dataset + "/test.tsv"
    payloads, _ = read_payloads_csv(payload_path)
    # payloads = payloads[:10]
    print("... success in loading %d samples from %s ..."%(len(payloads), input_args.dataset))
    
    begin_time = time.time()
    begin_time_str = time.strftime("%m-%d#%H-%M-%S", time.localtime())
    log_path = './logs/{}'.format(begin_time_str)
    os.mkdir(log_path)
    fbenign = open('{}/{}-{}#benign.log'.format(log_path, input_args.detector, input_args.dataset), 'a+')
    fexcept = open('{}/{}-{}#except.log'.format(log_path, input_args.detector, input_args.dataset), 'a+')
    fsummary = open('{}/{}-{}#summary.log'.format(log_path, input_args.detector, input_args.dataset), 'a+')
    fsuccess = open('{}/{}-{}#succes.log'.format(log_path, input_args.detector, input_args.dataset), 'a+')
    
    counter = {'total': len(payloads), 'success': 0, 'benign': 0, 'except': 0, 'failure': 0}
    cnt = 0
    for idx, payload in tqdm(enumerate(payloads)):
        run_res = evade(clsf, payload, input_args.max_rounds, input_args.round_size, input_args.timeout, input_args.random_engine, input_args.output_path)        
        if run_res['benign']:
            print(idx, payload, file=fbenign, flush=True)
            counter['benign'] = counter['benign'] + 1
        elif run_res['except']:
            print(idx, payload, run_res['except'], file=fexcept, flush=True)
            counter['except'] = counter['except'] + 1
        elif run_res['success']:
            print(idx, 'success', repr(run_res['min_payload']), file=fsuccess, flush=True)
            counter['success'] = counter['success'] + 1
        else:
            counter['failure'] = counter['failure'] + 1
        
        # break
    end_time = time.time()
    time_consume = end_time - begin_time
    
    asr = counter['success'] / (counter['success'] + counter['failure'])

    print("Total payloads: {}, Success: {}, Failure: {}, Benign: {}, Except: {}, AttackSuccessRate: {}".format(
        counter['total'], counter['success'], counter['failure'], counter['benign'], counter['except'], round(asr, 6)))
    print("Total payloads: {}, Success: {}, Failure: {}, Benign: {}, Except: {}".format(
        counter['total'], counter['success'], counter['failure'], counter['benign'], counter['except']), file=fsummary, flush=True)
    print("Total time consuming: {}h/{}m/{}s".format(round(time_consume/3600, 4),
                                                     round(time_consume/60, 4), round(time_consume, 4)))
    print("Total time consuming: {}h/{}m/{}s".format(round(time_consume/3600, 4),
                                                     round(time_consume/60, 4), round(time_consume, 4)), file=fsummary, flush=True)
    print("For detauil log, please see {}/".format(log_path))

    

if __name__ == '__main__':
    main()
