import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import pickle
import os
import soundfile as sf
from tqdm import tqdm
import torchaudio
from collections import defaultdict
import pandas as pd
import re
import html
import logging
from tqdm import tqdm
import pickle
import os

# refer to bert and mfcc
from input_embedding import AudioTextBatchFunction, BertForTextRepresentation
# tokenizer from the best model
from kobert import BertTokenizer

logger = logging.getLogger(__name__)


class DataSets(Dataset):
    """ Adapted from original multimodal transformer code"""

    def __init__(self, args, path, 
                 label_list,
                 split_type="train"):
        super(DataSets, self).__init__()
        
        # dataset info
        self.label2idx = dict(zip(label_list, range(len(label_list))))
        self.split_type = split_type        
        self.n_modalities = 2  # / text/ audio
        
        if not os.path.isfile(f'../processed/{args.model_name}/{self.split_type}.pkl'):
            

            # pretrained bert
            self.tokenizer, self.vocab = self.get_pretrained_model(args.bert_raw_text_path, 'etri')
            self.pad_idx = self.vocab['[PAD]']
            self.cls_idx = self.vocab['[CLS]']
            self.sep_idx = self.vocab['[SEP]']
            self.mask_idx = self.vocab['[MASK]']

            # get data
            self.audio, self.text, self.labels = self.get_data(path)





            # embedding
            self.mfcc_bert = AudioTextBatchFunction(args= args,
                                    pad_idx = self.pad_idx,
                                    cls_idx = self.cls_idx, 
                                    sep_idx = self.sep_idx,
                                    bert_args = args.bert_args,
                                    num_label_from_bert = len(label_list),
                                    device ='cpu'
                                   )

            self.audio_emb, self.audio_mask, self.text_emb, self.labels = self.mfcc_bert(self.audio, self.text, self.labels )
            os.makedirs(f'../processed/{args.model_name}', exist_ok=True)
            with open(f'../processed/{args.model_name}/{self.split_type}.pkl', 'wb') as f:
                pickle.dump({'audio':self.audio_emb,
                             'audio_mask':self.audio_mask,
                              'text':self.text_emb,
                              'labels':self.labels}, f)
        else:
            logger.info('loading saved files')
            with open(f'../processed/{args.model_name}/{self.split_type}.pkl', 'rb') as f:
                data = pickle.load(f)
            
            self.audio_emb, self.audio_mask, self.text_emb, self.labels = data['audio'],data['audio_mask'],data['text'],data['labels']
            
    def get_n_modalities(self):
        return self.n_modalities


    def __len__(self):
        return len(self.labels)
    
    
    def get_data(self, file_path):
        
        data = pd.read_pickle(file_path)
        
        text = data['Sentence']
        audio =  data['audio']
        

        label = [self.label2idx[l] for l in data['Emotion']]
        
        text = [self.normalize_string(sentence) for sentence in text]
        # Tokenize by Wordpiece tokenizer
        text = [self.tokenize(sentence) for sentence in text]
        
        # Change wordpiece to indices
        text = [self.tokenizer.convert_tokens_to_ids(sentence) for sentence in text]
        
        # as float32
        audio = np.array([a.astype(np.float32) for a in audio])
        # ------------------------guideline------------------------------------
        # naming as labels -> use to sampler
        # float32 is required for mfcc function in torchaudio
        #----------------------------------------------------------------------
        return audio, text, label
    
    def get_embedding(self):
        self.mfcc_bert(zip(self.audio, self.text, self.labels))
        
    def __getitem__(self, index):
        
        return self.audio_emb[index], self.audio_mask[index], self.text_emb[index], self.labels[index]
    
    @staticmethod
    def normalize_string(s):
        s = html.unescape(s)
        s = re.sub(r"[\s]", r" ", s)
        s = re.sub(r"[^a-zA-Z가-힣ㄱ-ㅎ0-9.!?]+", r" ", s)
        return s
    def tokenize(self, tokens):
        return self.tokenizer.tokenize(tokens)

    
    def get_pretrained_model(self, tokenizer_path, pretrained_type):
        # use etri tokenizer
        tokenizer = BertTokenizer.from_pretrained(
            tokenizer_path, do_lower_case=False)
        vocab = tokenizer.vocab


        return tokenizer, vocab
    
    
    
class Multimodal_Datasets(Dataset):
    def __init__(self, dataset_path, data='mosei_senti', split_type='train', if_align=False):
        super(Multimodal_Datasets, self).__init__()
        dataset_path = os.path.join(dataset_path, data+'_data.pkl' if if_align else data+'_data_noalign.pkl' )
        dataset = pickle.load(open(dataset_path, 'rb'))

        # These are torch tensors
        self.vision = torch.tensor(dataset[split_type]['vision'].astype(np.float32)).cpu().detach()
        self.text = torch.tensor(dataset[split_type]['text'].astype(np.float32)).cpu().detach()
        self.audio = dataset[split_type]['audio'].astype(np.float32)
        self.audio[self.audio == -np.inf] = 0
        self.audio = torch.tensor(self.audio).cpu().detach()
        self.labels = torch.tensor(dataset[split_type]['labels'].astype(np.float32)).cpu().detach()
        
        # Note: this is STILL an numpy array
        self.meta = dataset[split_type]['id'] if 'id' in dataset[split_type].keys() else None
       
        self.data = data
        
        self.n_modalities = 3 # vision/ text/ audio
    def get_n_modalities(self):
        return self.n_modalities
    def get_seq_len(self):
        return self.text.shape[1], self.audio.shape[1], self.vision.shape[1]
    def get_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]
    def get_lbl_info(self):
        # return number_of_labels, label_dim
        return self.labels.shape[1], self.labels.shape[2]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        X = (index, self.text[index], self.audio[index], self.vision[index])
        Y = self.labels[index]
        META = (0,0,0) if self.meta is None else (self.meta[index][0], self.meta[index][1], self.meta[index][2])
        if self.data == 'mosi':
            META = (self.meta[index][0].decode('UTF-8'), self.meta[index][1].decode('UTF-8'), self.meta[index][2].decode('UTF-8'))
        if self.data == 'iemocap':
            Y = torch.argmax(Y, dim=-1)
        return X, Y, META 