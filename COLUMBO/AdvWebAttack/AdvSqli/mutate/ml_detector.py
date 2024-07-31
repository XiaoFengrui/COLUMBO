import sys
sys.path.append('/home/ustc-5/XiaoF/AdvWebDefen/AdvSqli')
sys.path.append('/home/ustc-5/XiaoF/AdvWebDefen/GenerAT-main/DeBERTa')
from cnn.cnn_interface import CNNInterface
from lstm.lstm_interface import LSTMInterface
from generat.generat_interface import GENERATInterface
from generat.tools import build_argument_parser


def set_generat_args(args):
    args.task_name = 'adv-sst-2'
    args.tag = 'deberta-v3-large'
    args.model_config = '../../GenerAT-main/deberta-v3-large/rtd_large.json'
    args.init_generator = '../../GenerAT-main/deberta-v3-large/pytorch_model.generator.bin'
    args.init_discriminator = '../../GenerAT-main/deberta-v3-large/pytorch_model.bin'
    args.data_dir = '/home/ustc-5/XiaoF/AdvWebDefen/Dataset/SIK'
    args.output_dir = '../generat/output'

    return args

class MLDetector:
    def __init__(self, ml_alg, dataset, tr = None, ablation = None):
        self.ml_alg = ml_alg
        if ml_alg in ['CNN', 'cnn']:
            self.ml_interface = CNNInterface(dataset=dataset, tr = tr)
        elif ml_alg in ['LSTM', 'lstm']:
            self.ml_interface = LSTMInterface(dataset=dataset, tr = tr)
        elif ml_alg in ['GenerAT', 'generat', 'GENERAT']:
            parser = build_argument_parser()
            parser.parse_known_args()
            args = parser.parse_args()
            args = set_generat_args(args)
            self.ml_interface = GENERATInterface(args, dataset=dataset, tr = tr, ablation = ablation)
        self.thre = self.ml_interface.thre

    def get_result(self, payload):
        res = self.ml_interface.get_res(payload)
        return res
    
    def get_score(self, payload):
        score = self.ml_interface.get_score(payload)
        return score
    
    def get_thresh(self):
        return self.thre

def main():
    clsf = MLDetector('GenerAT', 'HPD')
    print(clsf.get_result(payload='/*!union*/ sELECt/*%^.**/password/*disgust **/-- '))
    print(clsf.get_result(payload='Admin or 1=1'))
    print(clsf.get_result(payload='adMMin or 1=1'))
    print(clsf.get_result(payload='admin /**/or 1=1'))
    print(clsf.get_result(payload='admin or 1=1 --+'))
    print(clsf.get_result(payload='1"admin or 1=1'))


if __name__ == '__main__':
    main()
