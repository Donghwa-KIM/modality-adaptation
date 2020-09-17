# utils
import logging
import numpy as np
# audio
import librosa
import torchaudio
import torch
# text
import torch.nn as nn
from pytorch_transformers.modeling_bert import BertPreTrainedModel
from pytorch_transformers.modeling_bert import BertConfig, BertModel

# init
logger = logging.getLogger(__name__)


class AudioTextBatchFunction:
    
    # batch function for pytorch dataloader
    def __init__(self,
                 args,
                 pad_idx, cls_idx, sep_idx,
                 bert_args,
                 num_label_from_bert,
                 device = 'cpu'
                 ):
        self.only_audio = args.do_audio
        self.device = device

        # related to audio--------------------
        self.max_len_a = args.max_len_for_audio
        self.n_mfcc = args.n_mfcc
        self.n_fft_size = args.n_fft_size
        self.sample_lr = args.sample_rate
        self.resample_lr = args.resample_rate

        self.audio2mfcc = torchaudio.transforms.MFCC(sample_rate=self.resample_lr,
                                        n_mfcc= self.n_mfcc,
                                        log_mels=False,
                                        melkwargs = {'n_fft': self.n_fft_size}).to(self.device)
        
        if not self.only_audio:
            # related to text--------------------
            self.max_len_t = bert_args.max_len
            self.pad_idx = pad_idx
            self.cls_idx = cls_idx
            self.sep_idx = sep_idx


            self.bert_config = BertConfig(args.bert_config_path)
            self.bert_config.num_labels = num_label_from_bert

            self.model = BertForTextRepresentation(self.bert_config).to(self.device)
            pretrained_weights = torch.load(args.bert_model_path
                                                , map_location=torch.device(self.device))
            self.model.load_state_dict(pretrained_weights, strict=False)
            self.model.eval()


        
    def __call__(self, audio, texts, label):
        #audio, texts, label = list(zip(*batch))
 
        if not self.only_audio:
            # Get max length from batch
            max_len = min(self.max_len_t, max([len(i) for i in texts]))
            texts = torch.tensor([self.pad_with_text([self.cls_idx] + text + [self.sep_idx], max_len) for text in texts])
            masks = torch.ones_like(texts).masked_fill(texts == self.pad_idx, 0)

            with torch.no_grad():
                # text_emb = last layer
                text_emb, cls_token = self.model(**{'input_ids': texts.to(self.device),
                                                   'attention_mask': masks.to(self.device)})
            
                audio_emb, audio_mask = self.pad_with_mfcc(audio)
            
            return audio_emb, audio_mask, text_emb, torch.tensor(label)
        else:
            audio_emb, audio_mask = self.pad_with_mfcc(audio)
            return audio_emb, audio_mask, None, torch.tensor(label)



    def pad_with_text(self, sample, max_len):
        diff = max_len - len(sample)
        if diff > 0:
            sample += [self.pad_idx] * diff
        else:
            sample = sample[-max_len:]
        return sample

    def pad_with_mfcc(self, audios): 
        max_len_batch = min(self.max_len_a, max([len(a) for a in audios]))
        audio_array = torch.zeros(len(audios), self.n_mfcc, max_len_batch).fill_(float('-inf')).to(self.device)
        logger.info('get mfcc')
        for ix, audio in enumerate(audios):
            audio_ = librosa.core.resample(audio, self.sample_lr, self.resample_lr)
            audio_ = torch.tensor(self.trimmer(audio_))
            mfcc = self.audio2mfcc(audio_.to(self.device))
            sel_ix = min(mfcc.shape[1], max_len_batch)
            audio_array[ix,:,:sel_ix] = mfcc[:,:sel_ix]
            if ix % 100 ==0: logger.info(f'{ix}//{len(audios)}')

        # (bat, n_mfcc, seq) -> (bat, seq, n_mfcc)
        padded_array = audio_array.transpose(2,1)
        
        # key masking
        # (batch, seq)
        key_mask = padded_array[:,:,0]
        key_mask = key_mask.masked_fill(key_mask != float('-inf'), 0).masked_fill(key_mask == float('-inf'),1).bool()
        
        # -inf -> 0.0
        padded_array = padded_array.masked_fill(padded_array == float('-inf'), float(0))
        return padded_array, key_mask
    
    def trimmer(self, audio):
        fwd_audio = []
        fwd_init = np.float32(0)
        for a in audio:
            if fwd_init!=np.float32(a):
                fwd_audio.append(a)

        bwd_init = np.float32(0)
        bwd_audio =[]
        for a in fwd_audio[::-1]:
            if bwd_init!=np.float32(a):
                bwd_audio.append(a)
        return bwd_audio[::-1]    
    
    
class BertForTextRepresentation(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForTextRepresentation, self).__init__(config)
        self.bert = BertModel(config)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.fc = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                masked_lm_labels=None, classification_label=None):
        bert_output = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                token_type_ids=token_type_ids,
                                head_mask=head_mask)
        return bert_output


    