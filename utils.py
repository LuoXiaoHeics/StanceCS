import spacy, os, numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from gensim.corpora import Dictionary as gensim_dico
import pandas as pd
import ast
nlp = spacy.load("en_core_web_sm")


def parse_processed_stance_dataset(domain, max_words):
    datasets = {}

    data_dir1 = 'VAST/vast_'+domain+'.csv'
    data = pd.read_csv(data_dir1)
    
    dico = gensim_dico()

    text = data['post']
    dico = gensim_dico()
    for l in text:
        t = nlp(l)
        tokens = [str(t[i]) for i in range(len(t)-1)]
        _ = dico.doc2bow(tokens, allow_update=True)
    summary = data['topic']
    for l in summary:
        u = ast.literal_eval(l)
        # tokens = ' '.join(u fsor u in summary)
        # tokens_list = []
        # tokens_list = tokens.split(' ')
        _ = dico.doc2bow(u, allow_update=True)
    dico.filter_extremes(no_below=2, keep_n=max_words)
    dico.compactify()

    X = []
    docid = -1
    for i,words in enumerate(zip(text,summary)):
        t,s = words
        us = ast.literal_eval(s)
        target = ' '.join(u for u in us)
        t = t+' '+target
        tokens = nlp(t)
        tokens_list = [str(tokens[i]) for i in range(len(tokens)-1)]
        count_list = dico.doc2bow(tokens_list, allow_update=False)
        docid += 1
        X.append((docid, count_list))
    print(docid)
    datasets[domain] = X
    return datasets,dico


def count_list_to_sparse_matrix(X_list, dico):
    ndocs = len(X_list)
    voc_size = len(dico.keys())


    X_spmatrix = sp.lil_matrix((ndocs, voc_size))
    for did, counts in X_list:
        for wid, freq in counts:
            X_spmatrix[did, wid]=freq

    return X_spmatrix.tocsr()


def get_dataset_path(domain_name, exp_type):
    prefix ='./dataset/'
    if exp_type == 'small':
        fname = 'labelled.review'
    elif exp_type == 'all':
        fname = 'all.review'
    elif exp_type == 'test':
        fname = 'unlabeled.review'

    return os.path.join(prefix, domain_name, fname)

def get_stance_dataset(max_words=5000, exp_type='train'):
    datasets, dico = parse_processed_stance_dataset(exp_type, max_words)
    print('parsed data ' +str(exp_type))
    L_s = datasets[exp_type]
    X_s = count_list_to_sparse_matrix(L_s,dico)
    X_s = np.array(X_s.todense())
    return X_s, dico




def spacy_seed_concepts(dico):
    """
    Returns concepts which belongs to proper noun, noun, adjective, or adverb parts-of-speech-tag category
    """
    seeds = []
    concepts = list(dico.values())
    tags = ['PROPN', 'NOUN', 'ADJ', 'ADV']

    for item in tqdm(concepts):
        if '_' not in item:
            doc = nlp(item)
            switch = 0
            for token in doc:
                if token.pos_ not in tags:
                    switch = 1
                    break
                else:
                    continue
                
            if switch == 0:
                seeds.append(item)
                
    return set(seeds)

def spacy_seed_concepts_list(concepts):
    """
    Returns concepts which belongs to proper noun, noun, adjective, or adverb parts-of-speech-tag category
    """
    seeds = []
    tags = ['PROPN', 'NOUN', 'ADJ', 'ADV']

    for item in concepts:
        if '_' not in item:
            doc = nlp(item)
            switch = 0
            for token in doc:
                if token.pos_ not in tags:
                    switch = 1
                    break
                else:
                    continue
                
            if switch == 0:
                seeds.append(item)
                
    return set(seeds)


