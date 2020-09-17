import torch
import os
import logging
from datasets import DataSets, Multimodal_Datasets
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import itertools
import numpy as np
import random

logger = logging.getLogger(__name__)




def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        
        
        
        
def get_data(args, label_list, dataset = None , split="train"):
    """ Adapted from original multimodal transformer code """
    data_path = os.path.join(args.data_path, f"{split}.pkl")
    
    return DataSets(args, data_path, label_list, split)




class ResultWriter:
    def __init__(self, dir):
        """ Save training Summary to .csv 
        input
            args: training args
            results: training results (dict)
                - results should contain a key name 'val_loss'
        """
        self.dir = dir
        self.hparams = None
        self.load()
        self.writer = dict()

    def update(self, args, **results):
        now = datetime.now()
        date = '%s-%s-%s %s:%s' % (now.year, now.month, now.day, now.hour, now.minute)
        self.writer.update({'date': date})
        self.writer.update(results)
        self.writer.update(vars(args))

        if self.hparams is None:
            self.hparams = pd.DataFrame(self.writer, index=[0])
        else:
            self.hparams = self.hparams.append(self.writer, ignore_index=True)
        self.save()

    def save(self):
        assert self.hparams is not None
        self.hparams.to_csv(self.dir, index=False)

    def load(self):
        path = os.path.split(self.dir)[0]
        if not os.path.exists(path):
            os.makedirs(path)
            self.hparams = None
        elif os.path.exists(self.dir):
            self.hparams = pd.read_csv(self.dir)
        else:
            self.hparams = None
            
            
def get_cosine_alphas(args, scheduler):
    lr_plot=[]
    for _ in range(int(args.num_epochs)):
        lr_plot.append(scheduler.get_lr()[0])
        scheduler.step()
    return list(reversed(lr_plot))
            
    
    
    
def draw_cm(args, label_list, cm):
    fig, ax = plt.subplots(figsize=(9, 8))
    font_p = "./font/HANBatang.ttf"
    fontprop = fm.FontProperties(fname=font_p, size=15)

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(label_list))

    ax = plt.gca()
    plt.xticks(tick_marks)
    ax.set_xticklabels(label_list, fontproperties=fontprop)
    plt.yticks(tick_marks)
    ax.set_yticklabels(label_list, fontproperties=fontprop)

    plt.xlim(-0.5, len(label_list)-0.5)
    plt.ylim(-0.5, len(label_list)-0.5)
    
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=12)
    plt.tight_layout()
    plt.ylabel('True', fontsize=12)
    plt.xlabel('Predict', fontsize=12)
    plt.savefig(f'../{args.cls_model_name}/{args.params_name}/test_result.png', dpi=300)
    
    
    
def benchmark_get_data(args, dataset, split='train'):
    alignment = 'a' if args.aligned else 'na'
    data_path = os.path.join(args.data_path, dataset) + f'_{split}_{alignment}.dt'
    if not os.path.exists(data_path):
        print(f"  - Creating new {split} data")
        data = Multimodal_Datasets(args.data_path, dataset, split, args.aligned)
        torch.save(data, data_path)
    else:
        print(f"  - Found cached {split} data")
        print(data_path)
        data = torch.load(data_path)
    return data
