from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from src.model.bart import BART, BART_finetune
from src.utils.tokenizer import CustomTokenizer
from src.utils.load_data import RIA
from src.utils import load_data
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from transformers import BartConfig
from pathlib import Path
import pandas as pd
import hydra
import os


@hydra.main(config_path = Path(__file__).parents[1]/'configs', config_name = 'config')
def train(cfg : DictConfig) -> None:
    def load_bart_config(cfg_ : DictConfig, tokenizer):
        cfg_ = OmegaConf.to_container(cfg_, resolve=True)
        cfg_ = cfg_['bart']

        bart_config_ = BartConfig(
            vocab_size = eval(cfg_.pop('vocab_size')), 
            pad_token_id = eval(cfg_.pop('pad_token_id')),
            bos_token_id= eval(cfg_.pop('bos_token_id')),
            eos_token_id = eval(cfg_.pop('eos_token_id')),
            return_dict=True,
            **cfg_
            )
        return bart_config_

    bart_parameters = cfg.bart
    parameters = cfg.parameters

    checkpoint_callback = ModelCheckpoint(
        dirpath = parameters.checkpoints.save_path,
        verbose = parameters.checkpoints.verbose,
        monitor = parameters.checkpoints.monitor,
        mode = parameters.checkpoints.mode
    )

    early_stop_callback = EarlyStopping(
        monitor = parameters.checkpoints.monitor,
        min_delta = parameters.checkpoints.min_delta,
        patience = parameters.checkpoints.patience,
        verbose = parameters.checkpoints.verbose,
        mode = parameters.checkpoints.mode
    )

    dataset_path = Path(__file__).parents[1]/f'data/{parameters.dataset.dataset_name}'
    if parameters.dataset.dataset_name == 'ria':
        if not os.path.exists(dataset_path):
            RIA(dataset_path/'ria.json.gz', parameters.dataset.n_rows, parameters.dataset.chunk_size).get_data()
        data = RIA.load_data(dataset_path)
        data_train, data_val = train_test_split(data, test_size = parameters.dataset.test_size)
    else:
        load_data.collect_gazeta()
        data_train = pd.read_csv(dataset_path/'gazeta_train.csv')
        data_val = pd.read_csv(dataset_path/'gazeta_val.csv')

    if parameters.tokenizer.train:
        tokenizer = CustomTokenizer()
        tokenizer.train(data['text'].values)
    else:
        tokenizer = CustomTokenizer.load_from_pretrained(
            parameters.tokenizer.max_length
            )

    bart_config = load_bart_config(bart_parameters, tokenizer)
    if parameters.model.get('pretrain') is not None:
        model = BART(
            bart_config = bart_config,
            parameters = parameters.model.pretrain.train_args,
            data_train = data_train,
            data_val = data_val,
            col_article = parameters.dataset.col_article,
            col_summary = parameters.dataset.col_summary,
            tokenizer = tokenizer
            )

        max_epochs = parameters.model.pretrain.train_args.num_epoch
        n_gpus = parameters.model.pretrain.train_args.n_gpus
    else:
        model = BART_finetune(
            bart_config = bart_config,
            parameters = parameters.model.finetune.train_args,
            data_train = data_train,
            data_val = data_val,
            col_article = parameters.dataset.col_article,
            col_summary = parameters.dataset.col_summary,
            tokenizer = tokenizer
        )
        max_epochs = parameters.model.finetune.train_args.num_epoch
        n_gpus = parameters.model.finetune.train_args.n_gpus

    trainer = Trainer(
        gpus = n_gpus,
        max_epochs = max_epochs, 
        callbacks = [early_stop_callback, checkpoint_callback], 
        )
    trainer.fit(model)

if __name__ == '__main__':
    train()