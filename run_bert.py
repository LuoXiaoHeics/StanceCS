# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import argparse
from distutils.log import error
import gc
import os
from numpy.core.fromnumeric import argsort
import torch
import logging
import random
import numpy as np
# from torch._C import half
from model import *
from tqdm import tqdm, trange
from transformers import BertConfig, BertTokenizer, XLNetConfig, XLNetTokenizer, WEIGHTS_NAME
from transformers import AdamW, Adafactor, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.distributed as dist
from tensorboardX import SummaryWriter
# from model import *
import glob
import json
import shutil
import re
from glue_utils import *

logger = logging.getLogger(__name__)
try:
    from apex import amp
except ImportError:
    amp = None
# ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig)), ())
ALL_MODELS = (
    'bert-base-uncased',
    'bert-large-uncased',
    'bert-base-cased',
    'bert-large-cased',
    'bert-base-multilingual-uncased',
    'bert-base-multilingual-cased',
    'bert-base-chinese',
    'bert-base-german-cased',
    'bert-large-uncased-whole-word-masking',
    'bert-large-cased-whole-word-masking',
    'bert-large-uncased-whole-word-masking-finetuned-squad',
    'bert-large-cased-whole-word-masking-finetuned-squad',
    'bert-base-cased-finetuned-mrpc',
    'bert-base-german-dbmdz-cased',
    'bert-base-german-dbmdz-uncased',
    'xlnet-base-cased',
    'xlnet-large-cased'
)



MODEL_CLASSES = {
    'bert_stance' : (BertConfig, BertStance, BertTokenizer),
}



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='model_output')
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task", default='all', type=str, required=True,
                        help="Task for stance detection")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--fix_tfm", default=0, type=int, help="whether fix the transformer params or not")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run testing.")
    parser.add_argument("--do_dev", action='store_true',
                        help="Whether to run dev.")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1.5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument('--eval_steps', type=int, default=100,
                        help="Evaluate valid set every X steps.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=100,
                        help="Log every X updates steps.")
    parser.add_argument("--scheduler", default="linear", type=str, choices=["linear", "constant", "inv_sqrt"])
    parser.add_argument("--optimizer", default="adam", type=str, choices=["adam", "adafactor"])
    parser.add_argument('--seed', type=int, default=56,
                        help="random seed for initialization")
    parser.add_argument("--no_cuda", action='store_true', default=False,
                        help="Avoid using CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Distributed learning")
    parser.add_argument('--fp16', default=False, action="store_true")
    # args = parser.parse_args()
    args = parser.parse_args(["--data_dir", "VAST", "--model_type", "bert_stance",              
                              '--model_name_or_path', 'bert-base-uncased',
                              "--do_lower_case",'--do_test','--do_train',
                              "--max_seq_length", '256', "--task", 'zero',
                              '--output_dir','output'])
    if args.fp16 and amp is None:
        print("No apex installed, fp16 not used.")
        args.fp16 = False
    return args


def train(args, train_dataset, model,sent_model, tokenizer):
    """ Train the model """
    tb_writer = SummaryWriter(args.output_dir)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    # del train_dataset
    # gc.collect()

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params'      : [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if args.optimizer == "adam":
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    elif args.optimizer == "adafactor":
        optimizer = Adafactor(optimizer_grouped_parameters, lr=args.learning_rate, scale_parameter=False,
                              relative_step=False)

    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    if args.scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=t_total)
    elif args.scheduler == "constant":
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)
    else:
        scheduler = get_inverse_sqrt_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    set_seed(args)  # For reproducibility (even between python 2 and 3)
    best_p=0.0
    should_stop = False
    for n, p in sent_model.bert.named_parameters():
        p.requires_grad = False
    for epoch in range(int(args.num_train_epochs)):
        epoch_iterator = tqdm(train_dataloader,desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            loss = torch.tensor(0, dtype=float).to(args.device)

            inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                          }
            sent_hidden = sent_model(**inputs)
            
            inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                          'sent_labels': batch[3],
                         'graph_feature':batch[4],
                         'sent_hidden': sent_hidden,
                          }
            logits,l  = model(**inputs)
            # loss with attention mask
            if args.n_gpu >1:
                loss += l.mean()
            else :
                loss += l

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward(retain_graph=True)

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                if global_step % args.logging_steps == 0 or global_step == 1:
                    # Log metrics
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logger.info("epoch: {:d}, step: {:d}, "
                                "loss: {:.4f}, lr: {:g}".format(epoch, global_step,
                                                                (tr_loss - logging_loss) / args.logging_steps,
                                                                scheduler.get_lr()[0]))
                    logging_loss = tr_loss

                if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    results = evaluate(args, model,sent_model, tokenizer, mode="dev")
                    logger.info("macro-f1 {:4f}".format(results['macro-f1']))
                    tb_writer.add_scalar('dev_best_f1', global_step)
                    if best_p < results['macro-f1']:
                        best_p = results['macro-f1']
                        if not os.path.exists(args.output_dir):
                            os.mkdir(args.output_dir)
                        model.save_pretrained(args.output_dir)
                        tokenizer.save_pretrained(args.output_dir)
                        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
                        logger.info("Saving best model checkpoint.")

                    tb_writer.add_scalar('eval_best_p', best_p, global_step)

                if 0 < args.max_steps < global_step:
                    should_stop = True

            if should_stop:
                break
        if should_stop:
            break
    del train_dataloader
    tb_writer.close()
    return best_p

def load_and_cache_examples(args, tokenizer, mode='train'):
    processor = StanceLoader()
    print('__________loading '+mode +' dataset__________')
    if mode == 'train':
        examples = processor.get_train_examples(args.data_dir)
        graph_feature = torch.tensor(np.load(open('VAST/sf_train' +'_5000.np', 'rb'), allow_pickle=True),dtype=torch.float32)
    elif mode == 'dev':
        examples = processor.get_dev_examples(args.data_dir,args.task)
        if args.task=='all':
            graph_feature = torch.tensor(np.load(open('VAST/sf_dev_5000.np', 'rb'), allow_pickle=True),dtype=torch.float32)
        else:
            graph_feature = torch.tensor(np.load(open('VAST/sf_'+args.task +'_dev_5000.np', 'rb'), allow_pickle=True),dtype=torch.float32)
    elif mode == 'test':
        examples = processor.get_test_examples(args.data_dir,args.task)
        if args.task=='all':
            graph_feature = torch.tensor(np.load(open('VAST/sf_test_5000.np', 'rb'), allow_pickle=True),dtype=torch.float32)
        else:
            graph_feature = torch.tensor(np.load(open('VAST/sf_'+args.task +'_test_5000.np', 'rb'), allow_pickle=True),dtype=torch.float32)
        # print('__________loading '+mode +' dataset__________')
    
    features = convert_examples_to_features(examples=examples, max_seq_length=args.max_seq_length, tokenizer=tokenizer)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,graph_feature)
    del features 
    gc.collect()
    return dataset

SMALL_POSITIVE_CONST = 1e-4
def compute_metrics_absa(preds, labels):
    errors = []
    hit_count, gold_count, pred_count = np.zeros(3), np.zeros(3), np.zeros(3)
    ts_precision, ts_recall, ts_f1 = np.zeros(3), np.zeros(3), np.zeros(3)

    for u in range(len(preds)):
        pred_count[preds[u]]+=1
        gold_count[labels[u]] +=1
        if preds[u] == labels[u]:
            hit_count[preds[u]]+=1
        else : errors.append(u)
    for i in range(3):
        n_ts = hit_count[i]
        n_g_ts = gold_count[i]
        n_p_ts = pred_count[i]
        ts_precision[i] = float(n_ts) / float(n_p_ts + SMALL_POSITIVE_CONST)
        ts_recall[i] = float(n_ts) / float(n_g_ts + SMALL_POSITIVE_CONST)
        ts_f1[i] = 2 * ts_precision[i] * ts_recall[i] / (ts_precision[i] + ts_recall[i] + SMALL_POSITIVE_CONST)
    macro_f1 = ts_f1.mean()

    n_tp_total = sum(hit_count)
    # TP + FN
    n_g_total = sum(gold_count)
    n_p_total = sum(pred_count)
    micro_p = float(n_tp_total) / (n_p_total + SMALL_POSITIVE_CONST)
    micro_r = float(n_tp_total) / (n_g_total + SMALL_POSITIVE_CONST)
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + SMALL_POSITIVE_CONST)

    scores = {'oppose-f1': ts_f1[0],'support-f1': ts_f1[1],'neutral-f1': ts_f1[2],'macro-f1': macro_f1, "micro-f1": micro_f1}
    print(scores)
    return scores,errors

def evaluate(args, model, sent_model, tokenizer, mode):
    eval_dataset = load_and_cache_examples(args, tokenizer, mode=mode)

    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataloader)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    eval_loss, eval_steps = 0.0, 0
    logit_list, label_list = [],[]
    ids = []
    model.eval()
    for batch in eval_dataloader:
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            eval_loss = torch.tensor(0, dtype=float).to(args.device)
            inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                          }
            sent_hidden = sent_model(**inputs)
            
            inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                          'sent_labels': batch[3],
                         'graph_feature':batch[4],
                         'sent_hidden': sent_hidden,
                          }
            ids.extend(batch[0])
            logits,l  = model(**inputs)
            eval_loss += l
            logit_list.append(torch.argmax(logits, axis=-1))
            label_list.append(batch[3])
        eval_steps += 1
    preds = torch.cat(logit_list, axis=0)
    labels =torch.cat(label_list, axis=0)

    result,errors = compute_metrics_absa(preds, labels)
    return result

def main():
    args = init_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # Setup CUDA
    torch.cuda.set_device(1)
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu =1
    
    print("GPU number is : ~~~~~~~~~~~~~~~~  "+ str(args.n_gpu))
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    set_seed(args)

    args.model_type = args.model_type.lower()

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          output_hidden_states=True)
    config.sent_number = 3
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case, cache_dir='./cache')

    if args.do_train:
        model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config, cache_dir='./cache')
        model.to(args.device)

        sent_model = BertSent_Encode.from_pretrained('SentiX_Base_Model')
        sent_model.to(args.device)

        if args.n_gpu >1:
            model = torch.nn.DataParallel(model)
        train_dataset = load_and_cache_examples(args, tokenizer, mode='train')
        train(args, train_dataset, model,sent_model, tokenizer)
        del train_dataset 
        gc.collect()
        del model
        torch.cuda.empty_cache()

    if args.do_test:
        args.model_type = args.model_type.lower()
        r = 0
        with open (args.output_dir+'/test_results.txt','w') as f:
            model = model_class.from_pretrained(args.output_dir)
            model.to(args.device)
            sent_model = BertSent_Encode.from_pretrained('SentiX_Base_Model')
            sent_model.to(args.device)
            results = evaluate(args, model,sent_model, tokenizer, 'test')
            f.write("results "+str(results)+'\n')
        del model
        torch.cuda.empty_cache()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
