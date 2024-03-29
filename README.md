# StanceCS

## Libraries
python==3.8.10

transformers==4.1.1

pytorch==1.10.0

## Common Sense
Run the code to train the module of RGCN and extract the common sense feature of the data:
```
python preprocess_graph.py
python train_and_extract_graph_features.py
python extract_graph_features.py
```

You can also download the trained features and data from https://drive.google.com/file/d/1HYiIVxTTHuqpWhGxqW8q6iFQQSXGNR05/view?usp=sharing.

The original dataset can be downloaded from https://github.com/emilyallaway/zero-shot-stance.

The ConceptGraph can be downloaded from https://drive.google.com/file/d/19klcp69OYEf29A_JrBphgkMVPQ9rXe1k/view.

## Sentiment
The code for training of SentiBERT can be found in https://github.com/12190143/SentiX.

## Stance Detection
```
python run_bert.py
```

Some of our code comes from https://github.com/declare-lab/kingdom.

## Cite
@article{luo2022exploiting,

  title={Exploiting Sentiment and Common Sense for Zero-shot Stance Detection},
  
  author={Luo, Yun and Liu, Zihan and Shi, Yuefeng and Zhang, Yue},
  
  journal={arXiv preprint arXiv:2208.08797},
  
  year={2022}
  
}
