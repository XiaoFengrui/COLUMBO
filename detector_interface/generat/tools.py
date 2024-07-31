import argparse
import numpy as np
import torch

import random
from collections import OrderedDict

import sys
import os
import importlib
from glob import glob
sys.path.append('/home/ustc-5/XiaoF/AdvWebDefen/GenerAT-main/DeBERTa/')
from training import get_args as get_training_args
from optims import get_args as get_optims_args
from data.example import ExampleSet, ExampleInstance, _truncate_segments
from apps.tasks import EvalData, Task
from apps.tasks.metrics import *
from data import DynamicDataset
from apps.tasks import get_task, load_tasks_new

def boolean_string(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'

class LoadTaskAction(argparse.Action):
    _registered = False

    def __call__(self, parser, args, values, option_string=None):
        setattr(args, self.dest, values)
        if not self._registered:
            load_tasks_new(args.task_dir)
            all_tasks = get_task()
            # print(all_tasks)
            if values == "*":
                for task in all_tasks.values():
                    parser.add_argument_group(title=f'Task {task._meta["name"]}', description=task._meta["desc"])
                return

            assert values.lower() in all_tasks, f'{values} is not registed. Valid tasks {list(all_tasks.keys())}'
            task = get_task(values)
            group = parser.add_argument_group(title=f'Task {task._meta["name"]}', description=task._meta["desc"])
            task.add_arguments(group)
            type(self)._registered = True

def build_argument_parser():
    parser = argparse.ArgumentParser(parents=[get_optims_args(), get_training_args()],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    load_tasks_new()

    # duplicate definition
    parser.add_argument('--detector', '-d', required=True, default='CNN', choices=['CNN', 'LSTM', 'GenerAT'], help='attack detector')
    parser.add_argument('--dataset', '-ds', required=True, default='HPD', choices=['HPD', 'SIK'], help='choose dataset')
    parser.add_argument('--guide', '-g', required=False, default='random', choices=[
                    'random', 'mcts'], help='guide mothod: mcts or random (default); mcts means Monte-Carlo Tree Search')
    parser.add_argument('--max_attempts', '-mat', required=False, default=10,
                    type=int, help='maximum number of attempts, default is 10; This parameter is for the entire attack process, no matter what attack method is used, the attack process is repeated *mat* times')
    parser.add_argument('--max_steps', '-mst', required=False, default=10, type=int,
                    help='this parameter plays a role in the attack process, a payload can be mutated at most *mst* times.')
    parser.add_argument("--max_rounds", "-r", default=1000, type=int, help="Maximum number of fuzzing rounds. Default: 1000") # steps
    parser.add_argument("--round_size", "-s", default=10, type=int, help="Fuzzing step size for each round (parallel fuzzing steps). Default: 20") # budget
    parser.add_argument("--training_ratio", "-tr", type=int, default=None, help="Training traio")
    parser.add_argument("--ablation_selection", "-as",  default=None, choices=['RTDGAT', 'RTD', 'GAT'], help="Ablation Selection, without which modules")

    ## Required parameters
    parser.add_argument("--task_dir",
                        default=None,
                        type=str,
                        required=False,
                        help="The directory to load customized tasks.")
    
    parser.add_argument("--task_name",
                        default='adv-sst-2',
                        type=str,
                        action=LoadTaskAction,
                        required=False,
                        help="The name of the task to train. To list all registered tasks, use \"*\" as the name, e.g. \n"
                             "\npython -m DeBERTa.apps.run --task_name \"*\" --help")

    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=False,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    # parser.add_argument("--adv_data_path",
    #                     default=None,
    #                     type=str,
    #                     required=False,
    #                     help="The input adv eval data file path. The path of `dev.json` file.")

    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=False,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")

    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--do_predict",
                        default=False,
                        action='store_true',
                        help="Whether to run prediction on the test set.")

    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")

    parser.add_argument("--predict_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for prediction.")

    parser.add_argument('--init_model',
                        type=str,
                        help="The model state file used to initialize the model weights.")

    parser.add_argument('--model_config',
                        type=str,
                        help="The config file of bert model.")

    parser.add_argument('--cls_drop_out',
                        type=float,
                        default=None,
                        help="The config file model initialization and fine tuning.")

    parser.add_argument('--tag',
                        type=str,
                        default='final',
                        help="The tag name of current prediction/runs.")

    parser.add_argument('--debug',
                        default=False,
                        type=boolean_string,
                        help="Whether to cache cooked binary features")

    parser.add_argument('--pre_trained',
                        default=None,
                        type=str,
                        help="The path of pre-trained RoBERTa model")

    parser.add_argument('--vocab_type',
                        default='gpt2',
                        type=str,
                        help="Vocabulary type: [spm, gpt2]")

    parser.add_argument('--vocab_path',
                        default=None,
                        type=str,
                        help="The path of the vocabulary")

    parser.add_argument('--cache_dir',
                        default=None,
                        type=str,
                        help="The path of the cache model")

    parser.add_argument('--vat_lambda',
                        default=0,
                        type=float,
                        help="The weight of adversarial training loss.")

    parser.add_argument('--vat_learning_rate',
                        default=1e-4,
                        type=float,
                        help="The learning rate used to update pertubation")

    parser.add_argument('--rtd_weight',
                        default=0,
                        type=float,
                        help="The weight of replaced token detenction loss.")

    parser.add_argument('--vat_init_perturbation',
                        default=1e-2,
                        type=float,
                        help="The initialization for pertubation")

    parser.add_argument('--vat_loss_fn',
                        default="symmetric-kl",
                        type=str,
                        help="The loss function used to calculate adversarial loss. It can be one of symmetric-kl, kl or mse.")

    parser.add_argument('--export_onnx_model',
                        default=False,
                        type=boolean_string,
                        help="Whether to export model to ONNX format.")

    parser.add_argument("--num_train_steps",
                        default=-1,
                        type=int,
                        help="Total training step. default -1 is not use, use epoch to determine the steps")

    return parser

def create_model(args, num_labels, model_class_fn):
    # Prepare model
    rank = getattr(args, 'rank', 0)
    init_model = args.init_model if rank < 1 else None
    model = model_class_fn(init_model, args.model_config, num_labels=num_labels, \
                           drop_out=args.cls_drop_out, \
                           pre_trained=args.pre_trained)
    if args.fp16:
        model = model.half()

    # logger.info(f'Total parameters: {sum([p.numel() for p in model.parameters()])}')
    # logger.info(f'Total discriminator parameters: {sum([p.numel() for p in model.discriminator.parameters()])}')
    return model

def predict_fn(logits):
    preds = np.argmax(logits, axis=-1)
    # labels = self.get_labels()
    return [int(p) for p in preds]

def metrics_fn(logits, labels):
    return OrderedDict(accuracy=metric_accuracy(logits, labels))

def example_to_feature(tokenizer, example, max_seq_len=512, rng=None, mask_generator=None,
                           ext_params=None,
                           **kwargs):
        if not rng:
            rng = random
        max_num_tokens = max_seq_len - len(example.segments) - 1 - 1
        segments = _truncate_segments([tokenizer.tokenize(s) for s in example.segments], max_num_tokens, rng)

        _tokens = ['[CLS]']
        type_ids = [0]
        for i, s in enumerate(segments):
            _tokens.extend(s)
            _tokens.append('[SEP]')
            type_ids.extend([(i + 1) % 2] * (len(s) + 1))
        # print('segments', example.segments, len(example.segments))
        # print('tokens', _tokens, len(_tokens))
        if mask_generator:
            token_ids = tokenizer.convert_tokens_to_ids(_tokens)
            tokens, lm_labels = mask_generator.mask_tokens(_tokens, rng)

            dis_labels = [0] * len(lm_labels)
            for i, s in enumerate(lm_labels):
                if s != 0:
                    dis_labels[i] = 1
            masked_token_ids = tokenizer.convert_tokens_to_ids(tokens)
            features = OrderedDict(input_ids=token_ids,
                                   masked_token_ids=masked_token_ids,
                                   type_ids=type_ids,
                                   position_ids=list(range(len(token_ids))),
                                   input_mask=[1] * len(token_ids),
                                   dis_labels=dis_labels,
                                   lm_labels=lm_labels)
        else:
            dis_labels = [0] * len(_tokens)

            token_ids = tokenizer.convert_tokens_to_ids(_tokens)
            features = OrderedDict(input_ids=token_ids,
                                   type_ids=type_ids,
                                   position_ids=list(range(len(token_ids))),
                                   input_mask=[1] * len(token_ids),
                                   dis_labels=dis_labels)

        for f in features:
            features[f] = torch.tensor(features[f] + [0] * (max_seq_len - len(token_ids)), dtype=torch.int)
        if example.label is not None:
            features['labels'] = torch.tensor(example.label, dtype=torch.int)
        return features

def calc_metrics(predicts, labels, eval_loss, eval_item, eval_results, args, name, prefix, steps, tag):
    assert len(predicts) == len(labels)
    tb_metrics = OrderedDict()
    result = OrderedDict()
    metrics_fn = eval_item.metrics_fn
    predict_fn = eval_item.predict_fn
    if metrics_fn is None:
        eval_metric = metric_accuracy(predicts, labels)
    else:
        metrics = metrics_fn(predicts, labels)
        result.update(metrics)
        critial_metrics = set(metrics.keys()) if eval_item.critial_metrics is None or len(
            eval_item.critial_metrics) == 0 else eval_item.critial_metrics
        eval_metric = np.mean([v for k, v in metrics.items() if k in critial_metrics])
    result['eval_loss'] = eval_loss
    result['eval_metric'] = eval_metric
    result['eval_samples'] = len(labels)
    if args.rank <= 0:
        output_eval_file = os.path.join(args.output_dir, "eval_results_{}_{}.txt".format(name, prefix))
        output_eval_file = os.path.join(args.output_dir, "eval_results_{}_{}.txt".format(name, prefix))
        with open(output_eval_file, 'w', encoding='utf-8') as writer:
            # logger.info("***** Eval results-{}-{} *****".format(name, prefix))
            for key in sorted(result.keys()):
                print("%s = %s"%(key, str(result[key])))
                # writer.write("%s = %s\n" % (key, str(result[key])))
                tb_metrics[f'{name}/{key}'] = result[key]
            print('fnr = %f'%(1-result['recall']))

        if predict_fn is not None:  # glue task 都有这个方法
            predict_fn(predicts, args.output_dir, name, prefix)  # 调用对应的方法，存提交用的文件
        else:
            output_predict_file = os.path.join(args.output_dir, "predict_results_{}_{}.txt".format(name, prefix))
            np.savetxt(output_predict_file, predicts, delimiter='\t')
            output_label_file = os.path.join(args.output_dir, "predict_labels_{}_{}.txt".format(name, prefix))
            np.savetxt(output_label_file, labels, delimiter='\t')

    if not eval_item.ignore_metric:
        eval_results[name] = (eval_metric, predicts, labels)
    _tag = tag + '/' if tag is not None else ''

    def _ignore(k):
        ig = ['/eval_samples', '/eval_loss']
        for i in ig:
            if k.endswith(i):
                return True
        return False
