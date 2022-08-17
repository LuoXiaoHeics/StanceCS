from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import random
import sys
from io import open
# from nltk.tokenize import word_tokenize
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers import AutoTokenizer,BartTokenizer
import pickle
logger = logging.getLogger(__name__)
import numpy as np
import torch 
import pandas as pd
import ast
import spacy
import torch.nn.functional as F

class InputExample():
    def __init__(self,d_tokens,targets,label = None,fil = None):
        self.d_tokens = d_tokens
        self.t_tokens = targets
        self.label = label
        self.fil = fil

class SentGloveFeatures(object):
    def __init__(self,tokens,embeddings,input_mask,label,domain):
        self.text_a = tokens
        self.input_mask = input_mask
        self.label = label
        self.domain = domain
        self.embeddings = embeddings

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label

class DataLoader__(object):
    """Base class for data converters for sequence classification data sets."""
    def get_train_examples(self, data_dirs):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dirs):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dirs):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()
        

class StanceLoader(DataLoader__):
    def get_train_examples(self, data_dirs):
        return self._create_examples(data_dirs=data_dirs, set_type='train')

    def get_dev_examples(self, data_dirs,task):
        return self._create_examples(data_dirs=data_dirs, set_type='dev',task=task)

    def get_test_examples(self, data_dirs,task):
        return self._create_examples(data_dirs=data_dirs, set_type='test',task=task)

    # def _create_examples(self, data_dirs, set_type, genre='Reviews'):
    #     input_examples = []
    #     if genre == 'Reviews':
    #         data_dir = os.path.join(data_dirs,'new_'+set_type+'.csv')
    #         data = pd.read_csv(data_dir)
    #         text = data['text']
    #         summary = data['summary']
    #         for id,d in enumerate(zip(text,summary)):
    #             t0,l = d
    #             t = ''
    #             targets = l.split(';')
    #             for u in targets[:-1]:
    #                 target, sent = u.split(',')
    #                 t+='<t_b> '+target + ' <t_e> '
    #                 t+=sent+' <t_s> '
    #             input_examples.append(InputExample(t0,t))
    #     return input_examples
        

    def _create_examples(self, data_dirs, set_type,task='all'):
        input_examples = []
        # tag2tagid = {'support': 0, 'oppose': 1, 'neutral': 2}
        if set_type=='train':
            data_dir = os.path.join(data_dirs,'vast_'+set_type+'.csv')
        else:
            if task == 'all':
                data_dir = os.path.join(data_dirs,'vast_'+set_type+'.csv')
            else: 
                data_dir = os.path.join(data_dirs,'vast_'+task+'_'+set_type+'.csv')
        data = pd.read_csv(data_dir)
        text = data['post']
        summary = data['topic']
        label = data['label']
        # filter = data['Sarc']
        a = 0
        for i,d in enumerate(zip(text,summary,label)):
            # t0,l,lab,f = d
            t0,l,lab = d
            l = ast.literal_eval(l)
            target = ' '.join(u for u in l)
            # input_examples.append(InputExample(t0,target,int(lab),fil = int(f)))
            input_examples.append(InputExample(t0,target,int(lab)))
        return input_examples
    

def statistics():
    data_dir = 'sentiment_lexicon_all.csv'
    data = pd.read_csv(data_dir)
    pos_set = set()
    neg_set = set()
    words = data['words']
    sentiment = data['sentiment']
    for i,u in enumerate(zip(words,sentiment)):
        if u[1] == 1:
            pos_set.add(u[0])
        else: neg_set.add(u[0])

    nlp = spacy.load("en_core_web_sm")
    data_dir1 = 'VAST/vast_test.csv'
    data = pd.read_csv(data_dir1)
    text = data['post']
    labels = data['label']
    matrix = np.zeros((3,3))
    positive,negative,neutral = 0,0,0
    sent_label = []
    for i,content in enumerate(zip(text,labels)):
        t,l = content
        p,n = 0,0
        s = nlp(t)
        tokens = [str(s[i]) for i in range(len(s)-1)]
        sent = -1
        for tok in tokens:
            if tok in pos_set:
                p+=1
            elif tok in neg_set:
                n+=1
        if p>n:
            positive+=1
            sent = 1
        elif p<n:
            negative+=1
            sent = 0
        else:
            neutral+=1
            sent = 2
        matrix[sent,l]+=1
        sent_label.append(sent*3+l)
    data['sent'] = sent_label

    input_examples = []
    text = data['post']
    summary = data['topic']
    label = data['label']
    filter = data['sent']
    for i,d in enumerate(zip(text,summary,label,filter)):
        t0,l,lab,f = d
        l = ast.literal_eval(l)
        target = ' '.join(u for u in l)
        input_examples.append(InputExample(t0,target,int(lab),fil = int(f)))
    return input_examples
    # print(matrix)
    # print(sent_label)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        tokens_a.pop()

def tokenizer_up(tokenizer):
    mapping = {
            'b': '<t_b>',
            'e': '<t_e>',
            'ts':'<t_s>'
    }
    # cur_num_tokens = tokenizer.vocab_size
    tokens_to_add = sorted(list(mapping.values()), key=lambda x:len(x), reverse=True)
    unique_no_split_tokens = tokenizer.unique_no_split_tokens
    sorted_add_tokens = sorted(list(tokens_to_add), key=lambda x:len(x), reverse=True)
        # for tok in sorted_add_tokens:
        #     print(self.tokenizer.convert_tokens_to_ids([tok])[0])
            # assert self.tokenizer.convert_tokens_to_ids([tok])[0]==self.tokenizer.unk_token_id
    tokenizer.unique_no_split_tokens = unique_no_split_tokens + sorted_add_tokens
    tokenizer.add_tokens(sorted_add_tokens)
    return tokenizer

def convert_examples_to_features(examples, max_seq_length,
                                 tokenizer,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.d_tokens)

        tokens_b = None
        if example.t_tokens:
            tokens_b = tokenizer.tokenize(example.t_tokens)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        # ms = [sequence_a_segment_id] * len(tokens)
        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)
            # ms += [0,0,1,0,0]*int(len(tokens_b)/5)+[0]
        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
            # ms = ms + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids
            # ms = [0]+ ms 
        # print(len(ms))
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
            # ms = ms + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label=example.label,
                              ))
    return features

class BartOTSAPipe():
    def __init__(self, tokenizer=None, max_length=128, gene_max_length = 128):
        # super(BartOTSAPipe, self).__init__()
        self.max_length = max_length
        self.gene_max_length = gene_max_length
        self.tokenizer = tokenizer
        # for tok in sorted_add_tokens:
        #     print(self.tokenizer.convert_tokens_to_ids([tok])[0])
            # assert self.tokenizer.convert_tokens_to_ids([tok])[0]==self.tokenizer.unk_token_id

    def encode_sentences1(self, data,  pad_to_max_length=True, ignore_pad_token_for_loss= True, return_tensors="pt"):
        input = self.tokenizer(
                list(data['text']),
                max_length=self.max_length,
                padding="max_length" if pad_to_max_length else None,
                truncation=True,
                return_tensors=return_tensors,
                add_prefix_space = True
            )
        # Setup the tokenizer for targets
        labels = self.tokenizer(list(data['golden_text']), max_length=self.gene_max_length, padding="max_length", truncation=True)
        input["labels"] = labels["input_ids"]
        decoder_input_ids = shift_tokens_right(torch.tensor(labels["input_ids"]),self.tokenizer.pad_token_id )
        # Shift the target ids to the right
        # shifted_target_ids = shift_tokens_right(encoded_dict['input_ids'], tokenizer.pad_token_id)
        if ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        encodings = {
            'input_ids': input['input_ids'],
            'attention_mask': input['attention_mask'],
            'decoder_input_ids': decoder_input_ids,
            'labels': labels["input_ids"],
        }
        
        return encodings
    
    def encode_sentences(self, source,target,  pad_to_max_length=True, ignore_pad_token_for_loss= True, return_tensors="pt"):
        input = self.tokenizer(
                (source),
                max_length=self.max_length,
                padding="max_length" if pad_to_max_length else None,
                truncation=True,
                return_tensors=return_tensors,
                # add_prefix_space = True
            )
        # Setup the tokenizer for targets
        labels = self.tokenizer(target, max_length=self.gene_max_length, padding="max_length", truncation=True)
        input["labels"] = labels["input_ids"]
        decoder_input_ids = shift_tokens_right(torch.tensor(labels["input_ids"]),self.tokenizer.pad_token_id )
        # Shift the target ids to the right
        # shifted_target_ids = shift_tokens_right(encoded_dict['input_ids'], tokenizer.pad_token_id)
        if ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        encodings = {
            'input_ids': input['input_ids'],
            'attention_mask': input['attention_mask'],
            'decoder_input_ids': decoder_input_ids,
            'labels': labels["input_ids"],
        }
        
        return encodings
        