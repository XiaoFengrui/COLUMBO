import torch
import time
import os
import sys
sys.path.append('/home/ustc-5/XiaoF/AdvWebDefen/AdvSqli')
from lstm.model import initilize_lstm_model
from lstm.tools import load_word2idx, predict

from enum import Enum

class LABEL(Enum):
    BENIGN = 0
    MALICIOUS = 1

# os.environ["CUDA_VISIBLE_DEVICES"] = '4'

class LSTMInterface():
    def __init__(self, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
                 dataset  = 'HPD',  
                 word2idx_path = '/home/ustc-5/XiaoF/AdvWebDefen/AutoSpear-main/classifiers/repos/lstm/word2idx.json', tr = None):
        super().__init__()
        if tr: # training ratio
            print('... load fewshot training ratio %d model ...'%tr)
            model_path = '/home/ustc-5/XiaoF/AdvWebDefen/AdvSqli/lstm/lstm_model_' + dataset + '_%d.pth'%tr
        else:             
            model_path = '/home/ustc-5/XiaoF/AdvWebDefen/AdvSqli/lstm/lstm_model_' + dataset + '.pth'
        # model_path = '/home/ustc-5/XiaoF/AdvWebDefen/AdvSqli/lstm/lstm_model_SIK.pth'
        self.device = device
        print('loading word2idx...')
        self.word2idx = load_word2idx(word2idx_path)
        self.model, _ = initilize_lstm_model(device, vocab_size=len(
            self.word2idx), embed_dim=300, dropout=0.5)
        self.model.load_state_dict(torch.load(
            model_path, map_location={'cuda': str(self.device)}))
        if dataset == 'HPD':
            self.thre = 0.5
        elif dataset == 'SIK':
            self.thre = 0.5
        
    def get_res(self, payload):
        score = predict(self.model, self.word2idx, self.device, payload)
        if score is None:
            score = 1.0
        res = LABEL(int(score > 0.5)).name
        return res
    
    def get_score(self, payload):
        # print('here')
        score = predict(self.model, self.word2idx, self.device, payload)
        if score is None:
            score = 1.0
        return score
 
if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model_path = './lstm_model.pth'
    # word2idx_path = '/home/ustc-5/XiaoF/AdvWebDefen/AutoSpear-main/classifiers/repos/lstm/word2idx.json'
    lstm = LSTMInterface()
    
    print('... Testing ...')
    t0 = time.time()
    score = lstm.get_score(payload='/*!union*/ sELECt/*%^.**/password/*disgust **/-- ')
    t1 = time.time()
    print('time cost: ', t1 - t0)
    
    score = lstm.get_score(payload="1%'   )    )    union all select null,null#")
    t2 = time.time()
    print('time cost: ', t2 - t1)
    
    score = lstm.get_score(payload='-8159 where 2793  =  2793 union all select 2793,2793,2793,2793,2793#')
    t3 = time.time()
    print('time cost: ', t3 - t2)
    
    score = lstm.get_score(payload='jimnez algora')
    t4 = time.time()
    print('time cost: ', t4 - t3)
    
    score = lstm.get_score(payload='"-8203""  )   union all select 6394,6394,6394,6394,6394--"')
    t5 = time.time()
    print('time cost: ', t5 - t4)
    
    score = lstm.get_score(payload="SELECT TOP 3 * FROM hall WHERE difficulty = 'outer'")
    t6 = time.time()
    print('time cost: ', t6 - t5)
    # print(lstm.get_score(payload='SELECT INSTR ( "W3Schools.com", "COM" )  AS MatchPosition;')) #0
    # print(lstm.get_score(payload='  SELECT MID ( "SQL Tutorial", -5, 5 )  AS ExtractString;')) #0
    # print(lstm.get_score(payload='40184#')) #0
    # print(lstm.get_score(payload='jimnez algora')) #0
    # print(lstm.get_score(payload='-1598))) union all select 9418,9418,9418,9418,9418,9418,9418# ')) #1
    # print(lstm.get_score(payload="SELECT TOP 3 * FROM hall WHERE difficulty = 'outer'")) #0
