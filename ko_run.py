import logging
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
    DistributedSampler,
)
import torch
import os
from tqdm import tqdm, trange
from sklearn.metrics import classification_report, confusion_matrix
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from utils import ResultWriter, get_cosine_alphas
import numpy as np

logger = logging.getLogger(__name__)


def train_modal_src(args, label_list, train_dataset, eval_dataset, model):
    # train(args, label_list, train_dataset, eval_dataset, model) 
    
    sampler = RandomSampler(train_dataset)
    
    args.batch_size = args.batch_size * max(1, args.n_gpu)
    
    train_loader = DataLoader(train_dataset, 
                              sampler=sampler,
                              batch_size=args.batch_size,
                              num_workers=0)
    
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    
    scheduler = CosineAnnealingWarmRestarts(optimizer, 
                                            args.num_epochs, T_mult=1, eta_min=0.0, last_epoch=-1)
    
    # modality learning rate
    alphas = get_cosine_alphas(args, scheduler) 
    alphas = np.array(alphas) 
    # total num_batches
    t_total = len(train_loader) * args.num_epochs *args.n_snapshot
    
    
    loss_fct = torch.nn.CrossEntropyLoss()
    
    logger.info(f"***** Running {args.cls_model_name} by train src *****")
    logger.info("  snapshot = %d Num Epochs = %d", args.n_snapshot, args.num_epochs)
    logger.info("  Total optimization steps = %d", t_total)

    #-------------------
    global_step = 0
    # cum_loss, current loss
    senti_tr_loss, modal_tr_loss, global_tr_loss = 0.0, 0.0, 0.0
    senti_logging_loss, modal_logging_loss, global_logging_loss = 0.0, 0.0, 0.0
    
    
    
    model.zero_grad()
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # saving folder name
    args.params_name = '-'.join([k+'='+ str(vars(args)[k]) for k in vars(args) if k not in [
                                                                    'data_name',
                                                                    'data_path',
                                                                    'bert_config_path',
                                                                    'bert_args',
                                                                    'bert_args_path',
                                                                    'bert_raw_text_path',
                                                                    'bert_model_path',
                                                                    'relu_dropout',
                                                                    'device',
                                                                    'n_gpu',
                                                                    'embed_dropout',
                                                                    'res_dropout',
                                                                    'out_dropout',
                                                                    'd_model',
                                                                    'n_modality',
                                                                    'batch_size',
                                                                    'clip',
                                                                    'optim',
                                                                    'num_epochs',
                                                                    'max_len_for_text',
                                                                    'hidden_size_for_bert',
                                                                    'max_len_for_audio',
                                                                    'sample_rate',
                                                                    'resample_rate',
                                                                    'n_fft_size',
                                                                    'n_mfcc',
                                                                    'logging_steps',
                                                                    'seed',
                                                                    'layers',
                                                                    'd_out',
                                                                    'num_heads',
                                                                    'attn_mask',
                                                                    'cls_model_name'
                                                                  ]])
    
    if not os.path.isdir(f"../{args.cls_model_name}/{args.params_name}"):
        os.makedirs(f"../{args.cls_model_name}/{args.params_name}")
    
    result_writer = ResultWriter(f"../{args.cls_model_name}/train_dev_results.csv")

    
    train_iterator = trange(0, int(args.num_epochs), desc="Epoch")
    for n in range(args.n_snapshot):
        best_val_loss = 1e8
        for i in train_iterator:
            epoch_iterator = tqdm(train_loader, desc="Iteration")
            for step, (audios, audio_mask, texts, labels) in enumerate(epoch_iterator):

                model.train()

                audios, audio_mask, texts = list(map(lambda x: x.to(args.device), [audios, audio_mask, texts]))
                labels = labels.squeeze(-1).long().to(args.device)
                modal_audio = torch.ones(audios.size(0), device=audios.device).long()
                modal_text = torch.zeros(texts.size(0), device=texts.device).long()
                modal_labels = torch.cat((modal_audio, modal_text), 0)


                inputs = {'x_audio': audios,
                  'x_text': texts,
                  'a_mask': audio_mask,
                  'alpha': alphas[i]} 

                senti_preds, modal_preds = model(**inputs)
                sentiLoss = loss_fct(senti_preds, labels.view(-1))
                modalLoss = loss_fct(modal_preds, modal_labels.view(-1))

                global_loss = sentiLoss + modalLoss

                # as a scalar
                if args.n_gpu > 1:
                    sentiLoss = sentiLoss.mean()
                    modalLoss = modalLoss.mean()
                    global_loss = global_loss.mean()

                # compute gradients
                global_loss.backward()

                # save the loss
                senti_tr_loss += sentiLoss.item()
                modal_tr_loss += modalLoss.item()
                global_tr_loss += global_loss.item()

                # clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

                # update
                optimizer.step()

                # init
                model.zero_grad()

                # cumulative step
                global_step += 1

                if global_step % args.logging_steps == 0:
                    logger.info("\n train loss : %.3f, senti_loss : %.3f, modal_loss : %.3f,\n \
                    lr : %.8f, alpha : %.8f", 
                                (global_tr_loss - global_logging_loss) / args.logging_steps,
                                (senti_tr_loss - senti_logging_loss) / args.logging_steps,
                                (modal_tr_loss - modal_logging_loss) / args.logging_steps,
                                scheduler.get_lr()[0], alphas[i]
                               )
                    global_logging_loss = global_tr_loss
                    senti_logging_loss = senti_tr_loss
                    modal_logging_loss = modal_tr_loss

            sent_val_loss, val_acc, val_macro_f1, _ = eval_src(args, label_list, eval_dataset, model)
            val_result = 'snapshot-{}-[{}/{}] val loss : {:.3f}, val acc : {:.3f}. val macro f1 : {:.3f}'.format(n, 
                global_step, t_total, sent_val_loss, val_acc, val_macro_f1
            )
            logger.info(val_result)

            scheduler.step()
            alpha=alphas[i]
            
            if sent_val_loss < best_val_loss:
                best_val_loss = sent_val_loss
                best_val_acc = val_acc
                best_val_macro_f1 = val_macro_f1


                torch.save(model, f"../{args.cls_model_name}/{args.params_name}/{n}-shot_model.pt")
                torch.save(args, f"../{args.cls_model_name}/{args.params_name}/args.pt")
                logger.info(f"  Saved {args.cls_model_name}-{args.params_name}")

                logger.info("  val loss : %.3f", sent_val_loss)
                logger.info("  best_val loss : %.3f", best_val_loss)

def train_src(args, label_list, train_dataset, eval_dataset, model):
    # train(args, label_list, train_dataset, eval_dataset, model) 
    
    sampler = RandomSampler(train_dataset)
    
    args.batch_size = args.batch_size * max(1, args.n_gpu)
    
    train_loader = DataLoader(train_dataset, 
                              sampler=sampler,
                              batch_size=args.batch_size,
                              num_workers=0)
    
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    
    scheduler = CosineAnnealingWarmRestarts(optimizer, 
                                            args.num_epochs, T_mult=1, eta_min=0.0, last_epoch=-1)
    

    
    # total num_batches
    t_total = len(train_loader) * args.num_epochs *args.n_snapshot
    
    
    loss_fct = torch.nn.CrossEntropyLoss()
    
    logger.info(f"***** Running {args.cls_model_name} by train src *****")
    logger.info("  snapshot = %d Num Epochs = %d", args.n_snapshot, args.num_epochs)
    logger.info("  Total optimization steps = %d", t_total)

    #-------------------
    global_step = 0
    # cum_loss, current loss
    senti_tr_loss = 0.0
    senti_logging_loss = 0.0
    
    
    
    model.zero_grad()
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # saving folder name
    args.params_name = '-'.join([k+'='+ str(vars(args)[k]) for k in vars(args) if k not in [
                                                                    'data_name',
                                                                    'data_path',
                                                                    'bert_config_path',
                                                                    'bert_args',
                                                                    'bert_args_path',
                                                                    'bert_raw_text_path',
                                                                    'bert_model_path',
                                                                    'relu_dropout',
                                                                    'device',
                                                                    'n_gpu',
                                                                    'embed_dropout',
                                                                    'res_dropout',
                                                                    'out_dropout',
                                                                    'd_model',
                                                                    'n_modality',
                                                                    'batch_size',
                                                                    'clip',
                                                                    'optim',
                                                                    'num_epochs',
                                                                    'max_len_for_text',
                                                                    'hidden_size_for_bert',
                                                                    'max_len_for_audio',
                                                                    'sample_rate',
                                                                    'resample_rate',
                                                                    'n_fft_size',
                                                                    'n_mfcc',
                                                                    'logging_steps',
                                                                    'seed',
                                                                    'layers',
                                                                    'd_out',
                                                                    'num_heads',
                                                                    'attn_mask',
                                                                    'cls_model_name'
                                                                  ]])
    
    if not os.path.isdir(f"../{args.cls_model_name}/{args.params_name}"):
        os.makedirs(f"../{args.cls_model_name}/{args.params_name}")
    
    result_writer = ResultWriter(f"../{args.cls_model_name}/train_dev_results.csv")

    
    train_iterator = trange(0, int(args.num_epochs), desc="Epoch")
    for n in range(args.n_snapshot):
        best_val_loss = 1e8
        for i in train_iterator:
            epoch_iterator = tqdm(train_loader, desc="Iteration")
            for step, (audios, audio_mask, texts, labels) in enumerate(epoch_iterator):

                model.train()

                audios, audio_mask, texts = list(map(lambda x: x.to(args.device), [audios, audio_mask, texts]))
                labels = labels.squeeze(-1).long().to(args.device)
                modal_audio = torch.ones(audios.size(0), device=audios.device).long()
                modal_text = torch.zeros(texts.size(0), device=texts.device).long()
                modal_labels = torch.cat((modal_audio, modal_text), 0)


                inputs = {'x_audio': audios,
                  'x_text': texts,
                  'a_mask': audio_mask} 

                senti_preds = model(**inputs)
                sentiLoss = loss_fct(senti_preds, labels.view(-1))


                # as a scalar
                if args.n_gpu > 1:
                    sentiLoss = sentiLoss.mean()

                # compute gradients
                sentiLoss.backward()

                # save the loss
                senti_tr_loss += sentiLoss.item()
  

                # clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

                # update
                optimizer.step()

                # init
                model.zero_grad()

                # cumulative step
                global_step += 1

                if global_step % args.logging_steps == 0:
                    logger.info("\n train loss : %.3f,\n \
                    lr : %.8f", 
                                (senti_tr_loss - senti_logging_loss) / args.logging_steps,
                                scheduler.get_lr()[0]
                               )
                    senti_logging_loss = senti_tr_loss

            sent_val_loss, val_acc, val_macro_f1, _ = eval_src(args, label_list, eval_dataset, model)
            val_result = 'snapshot-{}-[{}/{}] val loss : {:.3f}, val acc : {:.3f}. val macro f1 : {:.3f}'.format(n, 
                global_step, t_total, sent_val_loss, val_acc, val_macro_f1
            )
            logger.info(val_result)

            scheduler.step()
            
            if sent_val_loss < best_val_loss:
                best_val_loss = sent_val_loss
                best_val_acc = val_acc
                best_val_macro_f1 = val_macro_f1


                torch.save(model, f"../{args.cls_model_name}/{args.params_name}/{n}-shot_model.pt")
                torch.save(args, f"../{args.cls_model_name}/{args.params_name}/args.pt")
                logger.info(f"  Saved {args.cls_model_name}-{args.params_name}")

                logger.info("  val loss : %.3f", sent_val_loss)
                logger.info("  best_val loss : %.3f", best_val_loss)
            
def eval_src(args, label_list, eval_dataset, model):
    
    model.eval()
    
    sampler = SequentialSampler(eval_dataset)
    
    eval_loader = DataLoader(eval_dataset,
                             sampler=sampler, 
                             batch_size=args.batch_size,
                             num_workers=0)
            
    loss_fct = torch.nn.CrossEntropyLoss()
    
    senti_val_loss = 0.0
    val_acc, val_f1 = 0.0, 0.0
    
    total_y = []
    total_y_hat = []
                
    for val_step, (audios, audio_mask, texts, labels) in enumerate(tqdm(eval_loader, desc="Evaluating")):
        with torch.no_grad():
            
            audios, audio_mask, texts = list(map(lambda x: x.to(args.device), [audios, audio_mask, texts]))
            
            labels = labels.squeeze(-1).long().to(args.device)
            total_y += labels.tolist()
            
            inputs = {'x_audio': audios,
                      'x_text': texts,
                      'a_mask': audio_mask}   
            
            if args.adaptation:
                senti_preds, _ = model(**inputs)
            else:
                senti_preds = model(**inputs)
    
            sentiLoss = loss_fct(senti_preds, labels.view(-1))
                        
            
            # max => out: (value, index)
            y_max = senti_preds.max(dim=1)[1]
            total_y_hat += y_max.tolist()
            senti_val_loss += sentiLoss.item()
            
            
    # f1-score 계산
    cr = classification_report(total_y,
                               total_y_hat,
                               labels=list(range(len(label_list))),
                               target_names=label_list,
                               output_dict=True)
    # Get accuracy(micro f1)
    if 'micro avg' not in cr.keys():
        val_acc = list(cr.items())[-1][1]['f1-score']
    else:
        # If at least one of labels does not exists in mini-batch, use micro average instead
        val_acc = cr['micro avg']['f1-score']
    
    # macro f1
    val_macro_f1 = cr['macro avg']['f1-score']

    logger.info('***** Evaluation Results *****')
    f1_results = [(l, r['f1-score']) for i, (l, r) in enumerate(cr.items()) if i < len(label_list)]
    f1_log = "\n".join(["{} : {}".format(l, f) for l, f in f1_results])
    cm = confusion_matrix(total_y, total_y_hat)
    logger.info("\n***f1-score***\n" + f1_log + "\n***confusion matrix***\n{}".format(cm))
    
    sentiLoss /= (val_step + 1)
    
    return senti_val_loss, val_acc, val_macro_f1, (total_y_hat, cm, {l : f for l, f in f1_results})



def test_se(args, label_list, eval_dataset, models):
    
    for model in models:
        model.eval()
    
    sampler = SequentialSampler(eval_dataset)
    
    eval_loader = DataLoader(eval_dataset,
                             sampler=sampler, 
                             batch_size=args.batch_size,
                             num_workers=0)
            
    loss_fct = torch.nn.CrossEntropyLoss()
    
    senti_val_loss = 0.0
    val_acc, val_f1 = 0.0, 0.0
    
    total_y = []
    total_y_hat = []
                
    for val_step, (audios, audio_mask, texts, labels) in enumerate(tqdm(eval_loader, desc="Evaluating")):
        with torch.no_grad():
            
            audios, audio_mask, texts = list(map(lambda x: x.to(args.device), [audios, audio_mask, texts]))
            
            labels = labels.squeeze(-1).long().to(args.device)
            total_y += labels.tolist()
            
            inputs = {'x_audio': audios,
                      'x_text': texts,
                      'a_mask': audio_mask}   
            
            if args.adaptation:
                pred_lists = []
                for model in models:
                    senti_preds, _ = model(**inputs)
                    pred_lists.append(senti_preds.unsqueeze(0))
            else:
                pred_lists = []
                for model in models:
                    senti_preds = model(**inputs)
                    pred_lists.append(senti_preds.unsqueeze(0))
            senti_preds = torch.cat(pred_lists, 0).mean(dim = 0)
            sentiLoss = loss_fct(senti_preds, labels.view(-1))
                        
            
            # max => out: (value, index)
            y_max = senti_preds.max(dim=1)[1]
            total_y_hat += y_max.tolist()
            senti_val_loss += sentiLoss.item()
            
            
    # f1-score 계산
    cr = classification_report(total_y,
                               total_y_hat,
                               labels=list(range(len(label_list))),
                               target_names=label_list,
                               output_dict=True)
    # Get accuracy(micro f1)
    if 'micro avg' not in cr.keys():
        val_acc = list(cr.items())[-1][1]['f1-score']
    else:
        # If at least one of labels does not exists in mini-batch, use micro average instead
        val_acc = cr['micro avg']['f1-score']
    
    # macro f1
    val_macro_f1 = cr['macro avg']['f1-score']

    logger.info('***** Evaluation Results *****')
    f1_results = [(l, r['f1-score']) for i, (l, r) in enumerate(cr.items()) if i < len(label_list)]
    f1_log = "\n".join(["{} : {}".format(l, f) for l, f in f1_results])
    cm = confusion_matrix(total_y, total_y_hat)
    logger.info("\n***f1-score***\n" + f1_log + "\n***confusion matrix***\n{}".format(cm))
    
    sentiLoss /= (val_step + 1)
    
    return senti_val_loss, val_acc, val_macro_f1, (total_y_hat, cm, {l : f for l, f in f1_results})