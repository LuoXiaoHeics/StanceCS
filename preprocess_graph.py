from tqdm import tqdm
import numpy as np
import os.path, pickle
# from utils import obtain_all_seed_concepts
from utils_graph import conceptnet_graph, domain_aggregated_graph, subgraph_for_concept

import spacy, os, numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from gensim.corpora import Dictionary as gensim_dico
import pandas as pd

nlp = spacy.load("en_core_web_sm")

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

def get_seeds(max_words,type = 'all'):
    # First pass on document to build dictionary
    if type == 'all':
        data_dir1 = 'VAST/vast_train.csv'
        data_dir2 = 'VAST/vast_test.csv'
        data_dir3 = 'VAST/vast_dev.csv'
        data1 = pd.read_csv(data_dir1)
        data2 = pd.read_csv(data_dir2)
        data3 = pd.read_csv(data_dir3)
        data = pd.concat([data1,data2,data3],axis=0)
    else:
        data_dir1 = 'VAST/vast_'+type+'.csv'
        data = pd.read_csv(data_dir1)
    text = data['post']
    dico = gensim_dico()
    for l in text:
        t = nlp(l)
        tokens = [str(t[i]) for i in range(len(t)-1)]
        _ = dico.doc2bow(tokens, allow_update=True)
    # summary = data['text']
    # for l in summary:
    #     tokens = l.split(sep=';')
    #     tokens_list = []
    #     for tok in tokens[:-1]:
    #         ts, tfreq = tok.split(',')
    #         tokens_list = ts.split(' ')
    #     _ = dico.doc2bow(tokens_list, allow_update=True)
    dico.filter_extremes(no_below=2, keep_n=max_words)
    dico.compactify()
    seeds = spacy_seed_concepts(dico)
    return seeds

if __name__ == '__main__':
    
    bow_size = 5000
    
    print ('Extracting seed concepts from all domains.')
    all_seeds = get_seeds(bow_size,'all')
    
    print ('Creating conceptnet graph.')
    G, G_reverse, concept_map, relation_map = conceptnet_graph('conceptnet_english.txt')
    
    print ('Num seed concepts:', len(all_seeds))
    print ('Populating domain aggregated sub-graph with seed concept sub-graphs.')
    triplets, unique_nodes_mapping = domain_aggregated_graph(all_seeds, G, G_reverse, concept_map, relation_map)
    
    print ('Creating sub-graph for seed concepts.')
    concept_graphs = {}

    for node in tqdm(all_seeds, desc='Instance', position=0):
        concept_graphs[node] = subgraph_for_concept(node, G, G_reverse, concept_map, relation_map)
        
    # Create mappings
    inv_concept_map = {v: k for k, v in concept_map.items()}
    inv_unique_nodes_mapping = {v: k for k, v in unique_nodes_mapping.items()}
    inv_word_index = {}
    for item in inv_unique_nodes_mapping:
        inv_word_index[item] = inv_concept_map[inv_unique_nodes_mapping[item]]
    word_index = {v: k for k, v in inv_word_index.items()}
        
    print ('Saving files.')
        
    pickle.dump(all_seeds, open('utils/all_seeds.pkl', 'wb'))
    pickle.dump(concept_map, open('utils/concept_map.pkl', 'wb'))
    pickle.dump(relation_map, open('utils/relation_map.pkl', 'wb'))
    pickle.dump(unique_nodes_mapping, open('utils/unique_nodes_mapping.pkl', 'wb'))
    pickle.dump(word_index, open('utils/word_index.pkl', 'wb'))
    pickle.dump(concept_graphs, open('utils/concept_graphs.pkl', 'wb'))
    
    np.ndarray.dump(triplets, open('utils/triplets.np', 'wb'))        
    print ('Completed.')