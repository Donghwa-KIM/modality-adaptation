import argparse
import logging
import random
import os
from tqdm import tqdm, trange
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import itertools



# audio
import librosa
import torchaudio



# train
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler
)

# custom
from datasets import DataSets
from utils import *
from sklearn.metrics import classification_report, confusion_matrix
from run import train_modal_src, train_src, eval_src, test_se
from models import *

logger = logging.getLogger(__name__)



parser = argparse.ArgumentParser()
parser.add_argument(
    "--adaptation", action="store_true",  help="whether to use of adaptation"
)
parser.add_argument(
    "--late_fusion", action="store_true",  help="whether to use of late fusion (Transformer, etc.)"
)    
parser.add_argument(
    "--model_name", type=str, default="audio_text", help="name of the model to use (Transformer, etc.)"
)
parser.add_argument(
    "--data_name", type=str, default=None, help="dataset to use (default: vat)"
)
parser.add_argument(
    "--data_path", type=str, default="../data", help="path for storing the dataset"
)
parser.add_argument(
    "--bert_config_path", type=str, default="./kobert/best_model/bert_config.json", help="bert_config_path"
)
parser.add_argument(
    "--bert_args_path", type=str, default="./kobert/best_model/training_args.bin", help="bert_args_path"
)
parser.add_argument(
    "--bert_raw_text_path", type=str, default="./kobert/pretrained_model/etri/vocab.korean.rawtext.list", 
    help="bert_raw_text_path"
)

parser.add_argument(
    "--bert_model_path", type=str, default="./kobert/best_model/best_model.bin", 
    help="bert_model_path (pretrained & finetuned by a sentiment task)"
)


# Dropouts
parser.add_argument(
    "--attn_dropout_a", type=float, default=0.0, help="attention dropout (for audio)"
)
parser.add_argument(
    "--attn_dropout_t", type=float, default=0.0, help="attention dropout (for text)"
)
parser.add_argument("--relu_dropout", type=float, default=0.1, help="relu dropout")
parser.add_argument("--embed_dropout", type=float, default=0.1, help="embedding dropout")
parser.add_argument("--res_dropout", type=float, default=0.1, help="residual block dropout")
parser.add_argument("--out_dropout", type=float, default=0.0, help="output layer dropout")

# Architecture
parser.add_argument(
    "--layers", type=int, default=4, help="number of layers in the network (default: 5)"
)
parser.add_argument(
    "--d_model", type=int, default=40, help="dimension of layers in the network (default: 30)"
)
parser.add_argument(
    "--n_modality", type=int, default=2, help="number of modalities (default: 2)"
)
parser.add_argument(
    "--d_out", type=int, default=7, help="dimension of target dimension in the network (default: 7 for multi)"
)

parser.add_argument(
    "--num_heads",
    type=int,
    default=8,
    help="number of heads for the transformer network (default: 5)",
)
parser.add_argument(
    "--attn_mask",
    action="store_false",
    help="use attention mask for Transformer (default: true)",
)

# Tuning
parser.add_argument(
    "--batch_size", type=int, default=32, metavar="N", help="batch size (default: 24)"
)

parser.add_argument(
    "--clip", type=float, default=0.8, help="gradient clip value (default: 0.8)"
)
parser.add_argument(
    "--lr", type=float, default=1e-3, help="initial learning rate (default: 1e-3)"
)
parser.add_argument(
    "--n_snapshot", type=int, default=1, help="snapshot ensemble (default: 5)"
)

parser.add_argument("--num_epochs", type=int, default=5, help="number of epochs (default: 20)")

parser.add_argument(
    "--optim", type=str, default="Adam", help="optimizer to use (default: Adam)"
)

parser.add_argument("--max_len_for_text", default=64, type=int,
                        help="Maximum sequence length for text")
parser.add_argument("--hidden_size_for_bert", default=768, type=int,
                        help="hidden size used to a fine-tuned BERT")
parser.add_argument("--max_len_for_audio", default=400, type=int,
                        help="Maximum sequence length for audio")
parser.add_argument("--sample_rate", default=48000, type=int,
                        help="sampling rate for audio")
parser.add_argument("--resample_rate", default= 16000, type=int,
                        help="resampling rate to reduce audio sequence")
parser.add_argument("--n_fft_size", default=400, type=int,
                        help="time widnow for fourier transform")
parser.add_argument("--n_mfcc", default=40, type=int,
                        help="low frequency range (from 0 to n_mfcc)")

# Logistics
parser.add_argument(
    "--logging_steps", type=int, default=10, help="frequency of result logging (default: 30)"
)
parser.add_argument("--seed", type=int, default=1234, help="random seed")


args = parser.parse_args()


def main():
    # gpu assignment correspending to your PC 
    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    args.n_gpu = torch.cuda.device_count()        
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    # set seed
    set_seed(args)
    label_list=['공포', '놀람', '분노', '슬픔', '중립', '행복', '혐오']
    args.bert_args = torch.load(args.bert_args_path)
    
    train_dataset = get_data(args, label_list, args.data_name, "train")
    eval_dataset = get_data(args, label_list, args.data_name, "dev")
    test_dataset = get_data(args, label_list, args.data_name, "test")
    
    orig_d_a, orig_d_t= args.n_mfcc, args.hidden_size_for_bert

    args.cls_model_name = 'crossModal'
    args.d_out = len(label_list)
    
    
    audio_encoder = AudioEncoder(
    late_fusion = args.late_fusion,
    orig_d_a=orig_d_a,
    orig_d_t=orig_d_t,
    n_head=args.num_heads,
    n_layer=args.layers,
    d_model=args.d_model,
    emb_dropout=args.embed_dropout,
    attn_dropout=args.attn_dropout_a,
    relu_dropout=args.relu_dropout,
    res_dropout=args.res_dropout,
    out_dropout=args.out_dropout
    )

    text_encoder = TextEncoder(
        late_fusion = args.late_fusion,
        orig_d_a=orig_d_a,
        orig_d_t=orig_d_t,
        n_head=args.num_heads,
        n_layer=args.layers,
        d_model=args.d_model,
        emb_dropout=args.embed_dropout,
        attn_dropout=args.attn_dropout_t,
        relu_dropout=args.relu_dropout,
        res_dropout=args.res_dropout,
        out_dropout=args.out_dropout
    )

    senti_classifier = SentimentClassifier(args.d_model, args.d_out, args.out_dropout)
    modal_classifier = ModalityClassifier(args.d_model, args.n_modality, args.out_dropout)
    
    # train
    if args.adaptation:
        model = AdaptiveAudioTextClassifier(audio_encoder, text_encoder,
                                    senti_classifier, modal_classifier).to(device)
        train_modal_src(args, label_list, train_dataset, eval_dataset, model)

    else:
        model = AudioTextClassifier(audio_encoder, text_encoder,
                                    senti_classifier).to(device)
        train_src(args, label_list, train_dataset, eval_dataset, model)
    
    # test
    models = [torch.load( f"../{args.cls_model_name}/{args.params_name}/{n}-shot_model.pt") for n in range(args.n_snapshot)]
    tst_loss, tst_acc, tst_macro_f1, (total_y_hat, cm, cr) = test_se(args, label_list, test_dataset, models)
    print("loss : {} \nacc : {} \nf1 : {}".format(tst_loss, tst_acc, tst_macro_f1))
    test_result_writer = ResultWriter(f"../{args.cls_model_name}/test_results.csv")
    test_results = {
        'tst_loss': tst_loss,
        'tst_acc': tst_acc,
        'tst_macro_f1' : tst_macro_f1
    }
    
    # add f1 score for each class
    test_results.update(cr)

    # summary
    test_result_writer.update(args, **test_results)


    # confusion matrix
    draw_cm(args,label_list, cm) 
    
    
if __name__ == '__main__':
    main()