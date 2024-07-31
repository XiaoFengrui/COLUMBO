import torch
import time
import sys
sys.path.append('/home/ustc-5/XiaoF/AdvWebDefen/AdvSqli')
from cnn.model import initilize_cnn_model
from cnn.tools import load_word2idx, predict

from enum import Enum
import os

class LABEL(Enum):
    BENIGN = 0
    MALICIOUS = 1
    
# os.environ["CUDA_VISIBLE_DEVICES"] = '4'

class CNNInterface():
    def __init__(self, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
                 dataset  = 'HPD',  
                 word2idx_path = '/home/ustc-5/XiaoF/AdvWebDefen/AutoSpear-main/classifiers/repos/cnn/word2idx.json', 
                 tr = None):
        super().__init__()
        if tr: # training ratio
            print('... load fewshot training ratio %d model ...'%tr)
            model_path = '/home/ustc-5/XiaoF/AdvWebDefen/AdvSqli/cnn/cnn_model_' + dataset + '_%d.pth'%tr
        else:             
            model_path = '/home/ustc-5/XiaoF/AdvWebDefen/AdvSqli/cnn/cnn_model_' + dataset + '.pth'
        self.device = device
        print('loading word2idx...')
        self.word2idx = load_word2idx(word2idx_path)
        self.model, _ = initilize_cnn_model(device, vocab_size=len(
            self.word2idx), embed_dim=300, dropout=0.5)
        print(model_path)
        self.model.load_state_dict(torch.load(
            model_path, map_location={'cuda': str(self.device)}))
        self.thre = 0.5

    def get_res(self, payload):
        score = predict(self.model, self.word2idx, self.device, payload)
        if score is None:
            score = 1.0
        res = LABEL(int(score > self.thre)).name
        return res
    
    def get_score(self, payload):
        # print('here')
        score = predict(self.model, self.word2idx, self.device, payload)
        if score is None:
            score = 1.0
        return score
 
if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model_path = './cnn_model.pth'
    # word2idx_path = '/home/ustc-5/XiaoF/AdvWebDefen/AutoSpear-main/classifiers/repos/cnn/word2idx.json'
    dataset = 'HPD'
    cnn = CNNInterface(dataset=dataset)
    
    print('... Testing ...')
    t0 = time.time()
    score = cnn.get_score(payload='/*!union*/ sELECt/*%^.**/password/*disgust **/-- ')
    t1 = time.time()
    print('time cost: ', t1 - t0)
    
    score = cnn.get_score(payload="1%'   )    )    union all select null,null#")
    t2 = time.time()
    print('time cost: ', t2 - t1)
    
    score = cnn.get_score(payload='-8159 where 2793  =  2793 union all select 2793,2793,2793,2793,2793#')
    t3 = time.time()
    print('time cost: ', t3 - t2)
    
    score = cnn.get_score(payload='jimnez algora')
    t4 = time.time()
    print('time cost: ', t4 - t3)
    
    score = cnn.get_score(payload='"-8203""  )   union all select 6394,6394,6394,6394,6394--"')
    t5 = time.time()
    print('time cost: ', t5 - t4)
    
    score = cnn.get_score(payload="SELECT TOP 3 * FROM hall WHERE difficulty = 'outer'")
    t6 = time.time()
    print('time cost: ', t6 - t5)
    # print(cnn.get_score(payload='/*!union*/ sELECt/*%^.**/password/*disgust **/-- ')) #1
    # print(cnn.get_score(payload="1%'   )    )    union all select null,null")) #1
    # print(cnn.get_score(payload='-8159 where 2793  =  2793 union all select 2793,2793,2793,2793,2793#')) #1
    # print(cnn.get_score(payload='jimnez algora')) #0
    # print(cnn.get_score(payload='"-8203""  )   union all select 6394,6394,6394,6394,6394--"')) #1
    # print(cnn.get_score(payload="SELECT TOP 3 * FROM hall WHERE difficulty = 'outer'")) #0
