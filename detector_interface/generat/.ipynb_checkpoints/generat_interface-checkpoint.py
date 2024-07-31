import sys
import os
sys.path.append('/home/ustc-5/XiaoF/AdvWebDefen/GenerAT-main/DeBERTa')
# print(sys.path)
# from apps.models import GenerativeAdversarialTrainingModel
from deberta import tokenizers, load_vocab
from apps.tasks import load_tasks, get_task
from data.example import ExampleSet, ExampleInstance, _truncate_segments
from apps.tasks import EvalData, Task
from data import DynamicDataset
from data import DistributedBatchSampler, SequentialSampler, BatchSampler, AsyncDataLoader
from training import batch_to
from apps._utils import merge_distributed

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
sys.path.append('/home/ustc-5/XiaoF/AdvWebDefen/AdvSqli')
from generat.tools import build_argument_parser, create_model, example_to_feature, predict_fn, metrics_fn, calc_metrics

from enum import Enum
from collections import OrderedDict
from collections.abc import Sequence
from tqdm import tqdm
import time

# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
# num_gpus = torch.cuda.device_count()
# print(f"Number of available GPUs: {num_gpus}")
# torch.cuda.set_device(3)

class LABEL(Enum):
    BENIGN = 0
    MALICIOUS = 1
    
import pynvml
def get_gpu_memory_info():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(2)  # Assuming you're using GPU 0
    pid = os.getpid()
    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    
    for process in processes:
        if process.pid == pid:
            print(f"Current process GPU memory usage: {process.usedGpuMemory / (1024 ** 2):.2f} MB")
            return
    print('... Not Found Process ...')

class GENERATInterface():
    def __init__(self, args, device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu'), 
                 dataset = 'SIK', 
                 vocab_path = '/home/ustc-5/XiaoF/AdvWebDefen/GenerAT-main/deberta-v3-large/spm.model',
                 vocab_type = 'spm',
                 tr = None,
                 ablation = None):
        super().__init__()
        if tr:
            print('... load fewshot training ratio %d model ...'%tr)
            model_path = '/home/ustc-5/XiaoF/AdvWebDefen/AdvSqli/generat/generat_model_' + dataset + '_%d.bin'%tr
        elif ablation:
            print('... load model w/o %s module ...'%ablation)
            model_path = '/home/ustc-5/XiaoF/AdvWebDefen/AdvSqli/generat/generat_model_' + dataset + '_wo%s.bin'%ablation
        else:
            model_path = '/home/ustc-5/XiaoF/AdvWebDefen/AdvSqli/generat/generat_model_' + dataset + '.bin'
        self.args = args
        self.device = device
        # exit(0)
        self.tokenizer = tokenizers[vocab_type](vocab_path)
        # task 是 register 维护的 task_name 到对应 task dataset 的字典
        self.task = get_task('adv-sst-2')(tokenizer=self.tokenizer, args=self.args, max_seq_len=self.args.max_seq_length,
                                        data_dir=self.args.data_dir)

        print('... loading model ...')
        model_class_fn = self.task.get_model_class_fn()
        self.model = create_model(args, len(LABEL), model_class_fn)
        self.model.to(self.device)
        state_dict = torch.load(model_path, map_location = self.device)  # 或者 'cuda' 如果你想在 GPU 上加载
        self.model.discriminator.load_state_dict(state_dict)
        self.thre = 0.5

    def get_feature_fn(self, max_seq_len=512, mask_gen=None):
        def _example_to_feature(example, rng=None, ext_params=None, **kwargs):
            return example_to_feature(self.tokenizer, example, max_seq_len=max_seq_len, \
                                            rng=rng, mask_generator=mask_gen, ext_params=ext_params, **kwargs)

        return _example_to_feature

    def predict_batch(self, max_len=512, prefix=None, tag=None, steps=None):
        eval_data = self.task.test_data(max_seq_len=max_len)
        print(eval_data)
        prefix = self.args.tag
        prefix = f'{tag}_{prefix}' if tag is not None else prefix

        eval_results = OrderedDict()
        eval_metric = 0
        no_tqdm = (True if os.getenv('NO_TQDM', '0') != '0' else False) or args.rank > 0
        ort_session = None
        for eval_item in eval_data:
            name = eval_item.name
            eval_sampler = SequentialSampler(len(eval_item.data))
            batch_sampler = BatchSampler(eval_sampler, args.eval_batch_size) # batch size: 32
            # batch_sampler = DistributedBatchSampler(batch_sampler, rank=args.rank, world_size=args.world_size)
            eval_dataloader = DataLoader(eval_item.data, batch_sampler=batch_sampler, num_workers=args.workers)
            self.model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            predicts = []
            labels = []
            # print('+++++ here +++++')
            # print(eval_item.data.corpus[0])
            for batch in tqdm(eval_dataloader):
                # print(batch)
                _batch = batch.copy()
                batch = batch_to(batch, self.device)
                
                if ort_session is None:
                    with torch.no_grad():
                        output = self.model(**batch)
                logits = output['logits'].detach()
                tmp_eval_loss = output['loss'].detach()
                if 'labels' in output:
                    label_ids = output['labels'].detach().to(self.device)
                else:
                    label_ids = batch['labels'].to(self.device)
                # get_gpu_memory_info()
                predicts.append(logits)
                labels.append(label_ids)
                eval_loss += tmp_eval_loss.mean().item()
                input_ids = batch['input_ids']
                nb_eval_examples += input_ids.size(0)
                nb_eval_steps += 1
                # break

            eval_loss = eval_loss / nb_eval_steps
            predicts = merge_distributed(predicts, len(eval_item.data))
            labels = merge_distributed(labels, len(eval_item.data))
            # 这里 calc_metrics 的时候存了 sumbit prediction tsv 文件
            if isinstance(predicts, Sequence):
                for k, pred in enumerate(predicts):
                    calc_metrics(pred.detach().cpu().numpy(), labels.detach().cpu().numpy(), eval_loss, eval_item,
                                eval_results, args, name + f'@{k}', prefix, steps, tag)
            else:
                # print(predicts)
                calc_metrics(predicts.detach().cpu().numpy(), labels.detach().cpu().numpy(), eval_loss, eval_item,
                            eval_results, args, name, prefix, steps, tag)

        return eval_results
    
    def predict(self, model, payload, tokenizer, max_len=512, **kwargs):
        examples = ExampleSet([ExampleInstance((payload,))])
        t0 = time.time()
        ds = [EvalData('sst2', examples,
                        metrics_fn=metrics_fn, predict_fn=predict_fn)]
        for d in ds:
            _size = len(d.data)
            d.data = DynamicDataset(d.data, feature_fn=self.get_feature_fn(max_seq_len=max_len), dataset_size=_size, **kwargs)
        
        t1 = time.time()
        # print('Constructing Data Class Cost: ', t1 - t0)
        
        for item in ds:
            # print(item.data.corpus[0])
            # print(item.data[0])
            # print(self.args.workers)
            dataloader = DataLoader(item.data, batch_size=1, num_workers=1)
            model.eval()
            predicts = []
            t_0 = time.time()
            for batch in dataloader:
                batch = batch_to(batch, self.device)
                # print(self.device)
                with torch.no_grad():
                    output = model(**batch)
                logits = output['logits']
                predicts.append(logits)
            # print('Predicting Cost: ', time.time() - t_0, '\n')
        # print(predicts)
        logits = predicts[0].detach()
        probs = F.softmax(logits, dim=1).squeeze(dim=0)
        # print(logits)
        t2 = time.time()
        # print('Predicting Cost: ', t2 - t1, '\n')
        return probs[1].item()

    def get_res(self, payload):
        score = self.predict(self.model, payload, self.tokenizer)
        if score is None:
            score = 1.0
        res = LABEL(int(score > 0.5)).name
        return res
    
    def get_score(self, payload):
        score = self.predict(self.model, payload, self.tokenizer)
        if score is None:
            score = 1.0
        return score
 
if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model_path = './cnn_model.pth'
    # word2idx_path = '/home/ustc-5/XiaoF/AdvWebDefen/AutoSpear-main/classifiers/repos/cnn/word2idx.json'


    parser = build_argument_parser()
    parser.parse_known_args()
    args = parser.parse_args()

    generat = GENERATInterface(args, dataset='SIK')
    t = time.time()
    generat.predict_batch()
    print(time.time() - t)
    print('... Testing ...')
    t0 = time.time()
    score = generat.get_score(payload='/*!union*/ sELECt/*%^.**/password/*disgust **/-- ')
    t1 = time.time()
    print('time cost: ', t1 - t0)
    
    score = generat.get_score(payload="1%'   )    )    union all select null,null#")
    t2 = time.time()
    print('time cost: ', t2 - t1)
    
    score = generat.get_score(payload='-8159 where 2793  =  2793 union all select 2793,2793,2793,2793,2793#')
    t3 = time.time()
    print('time cost: ', t3 - t2)
    
    score = generat.get_score(payload='jimnez algora')
    t4 = time.time()
    print('time cost: ', t4 - t3)
    
    score = generat.get_score(payload='"-8203""  )   union all select 6394,6394,6394,6394,6394--"')
    t5 = time.time()
    print('time cost: ', t5 - t4)
    
    score = generat.get_score(payload="SELECT TOP 3 * FROM hall WHERE difficulty = 'outer'")
    t6 = time.time()
    print('time cost: ', t6 - t5)
    
    # print(generat.get_score(payload='/*!union*/ sELECt/*%^.**/password/*disgust **/-- ')) #1
    # print(generat.get_score(payload="1%'   )    )    union all select null,null#")) #1
    # print(generat.get_score(payload='-8159 where 2793  =  2793 union all select 2793,2793,2793,2793,2793#')) #1
    # print(generat.get_score(payload='jimnez algora')) #0
    # print(generat.get_score(payload='"-8203""  )   union all select 6394,6394,6394,6394,6394--"')) #1
    # print(generat.get_score(payload="SELECT TOP 3 * FROM hall WHERE difficulty = 'outer'")) #0
