# Huawei-project
This repository contains code for pre-training and fine-tuning BART for headline generation and summary generation.
NLP course template is a summary of what I have done.

## Usage

### Notebook API
Check example notebook notebooks/Headline_generation.ipynb.
### Config API
Here is a example for config API. All yo need to do is to change your bart config and parameters config, and then run train.py script.
```yaml
#configs/bart/bart.yaml
bart:
  vocab_size: tokenizer.get_vocab_size() #you can leave these steps unchanged
  pad_token_id: tokenizer.token_to_id("<pad>")
  bos_token_id: tokenizer.token_to_id("<s>")
  eos_token_id: tokenizer.token_to_id("</s>")
  encoder_layers: 6
  decoder_layers: 6
  activation_function: gelu 
  dropout: 0.1
  #additionally, you can pass parameters like dim_size, embedding_size and etc...
```
```yaml
#configs/parameters/parameters.yaml
checkpoints:
  save_path: model/checkpoints #path to checkpoint
  monitor: bleu #metric to monitor
  mode: max
  min_delta: 0.05
  patience: 2
  verbose: true

dataset:
  dataset_name: ria #ria, gazeta
  col_article: text #column name of the article
  col_summary: title #column name of the summary
  n_rows: 500000 #number of rows to take
  chunk_size: 10000
  test_size: 0.02

tokenizer:
  train: false #whether to train optimizer
  max_length: 750

model:
  finetune: # finetune or pretrain
    #additionally you can pass pretrained_path
    #pretrained_path: path of the pretrained model
    train_args:
      #parameters of the optimizer and trainer
      n_gpus: 1
      lr: 1e-5
      max_lr: 1e-4
      pct_start: 0.06
      num_epoch: 5
      batch_size: 8
      acc_step: 16
      max_length: ${tokenizer.max_length}
```
And then run it :
```
python src/train.py
```

## Inference
You can download finetuned model using gdown or use your own.
```
gdown https://drive.google.com/uc?id=1-pl-7i9a7QZrTNybZvaicIVmlcUbagMH
```
Then you can generate summaries for the specific text using API.

```python
from src.model.bart.finetune_model import generate_summary

#config - bart config
#path - path to the finetuned model
#text - text of the article
summary = generate_summary(config, path, text, max_length = 750, num_beams = 5)
```

## Code structure
 ```
 ├── NLP_Course_Template.pdf
├── README.md
├── configs
│   ├── bart
│   │   └── bart.yaml
│   ├── config.yaml
│   └── parameters
│       └── parameters.yaml
├── data
│   ├── gazeta_test.jsonl
│   ├── gazeta_train.jsonl
│   ├── gazeta_val.jsonl
│   └── ria.json.gz
├── model
│   └── tokenizer
│       ├── merges.txt
│       └── vocab.json
├── notebooks
│   └── Headline_generation.ipynb
├── requirements.txt
└── src
    ├── __init__.py
    ├── inference.py
    ├── loaders
    │   ├── __init__.py
    │   ├── finetune_loader.py
    │   └── pretrain_loader.py
    ├── model
    │   ├── __init__.py
    │   └── bart
    │       ├── __init__.py
    │       ├── finetune_model.py
    │       └── pretrain_model.py
    ├── train.py
    └── utils
        ├── __init__.py
        ├── load_data.py
        └── tokenizer.py
 ```
## Results
GAZETA dataset:

|             | BLEU   | ROUGE-2 | ROUGE-L |
|-------------|--------|---------|---------|
| n lead rows | 0.4341 | 0.1122  | 0.2264  |
| SummaRunner | 0.4145 | 0.1093  | 0.2229  |
| BART        | 0.3851 | 0.0944  | 0.2085  |

RIA dataset:

|             | BLEU   | ROUGE-2 | ROUGE-L |
|-------------|--------|---------|---------|
| n lead rows | 0.2025 | 0.1013  | 0.1618  |
| BART        | 0.5043 | 0.1999  | 0.3921  |

Svpressa dataset:

|             | BLEU   | ROUGE-2 | ROUGE-L |
|-------------|--------|---------|---------|
| n lead rows | 0.1563 | 0.0761  | 0.1488  |
| BART        | 0.1872 | 0.0663  | 0.1692  |

## Samples
**predicted**:  большая часть детей, которых граждане сша пытались вывезти из гаити в доминиканской приют,
не являются сиротами    
**reference**: большинство детей, которых пытались увезти в сша из гаити, не сироты  

**predicted**:  тимошенко надеется, что в случае победы на выборах президента будет работать в ее команде премьер-министр украины  
**reference**: луценко будет работать в команде тимошенко, если она победит в выборах  

**predicted**:  леверкузенский "байер" вернулся на первое место в чемпионате германии по футболу "фрайбург" 3:1"  
**reference**: "байер" вернулся в лидеры чемпионата германии по футболу  