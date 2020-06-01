# Huawei-project
This repository contains code for pre-training and fine-tuning BART, as well as code for collecting data from web pages.
NLP course template is a summary of what I have done.
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
| BART        | 0.3866 | 0.1897  | 0.3331  |

Svpressa dataset:

|             | BLEU   | ROUGE-2 | ROUGE-L |
|-------------|--------|---------|---------|
| n lead rows | 0.1563 | 0.0761  | 0.1488  |
| BART        | 0.1872 | 0.0663  | 0.1692  |

## Architectures
* BART
* SummaRuNNer
* LEAD-N
