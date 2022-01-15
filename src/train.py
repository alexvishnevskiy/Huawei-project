from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from src.model.bart import BART, BART_finetune
from src.utils.tokenizer import CustomTokenizer
from src.utils.load_data import RIA
from src.utils import load_data
from transformers import BartConfig
from pytorch_lightning import Trainer
import pandas as pd
from box import Box
import os


def train(args):
    # args = {
    #     'batch_size': BATCH_SIZE,
    #     'max_length': 300,
    #     'lr': 0.0001,
    # }
    args = Box(args)

    checkpoint_callback = ModelCheckpoint(
        dirpath = args.save_path,
        verbose=True,
        monitor='avg_bleu',
        mode='max'
    )

    early_stop_callback = EarlyStopping(
        monitor='avg_bleu',
        min_delta=0.00,
        patience=args.patience,
        verbose=False,
        mode='max'
    )

    if args.dataset_name == 'ria':
        if not os.path.exists('data/ria'):
            load_data.collect_data(args.n_rows, args.chunk_size)
        data = RIA.load_data('data/ria')
        data_train, data_val = train_test_split(data, test_size=args.test_size)
    else:
        data_train = pd.read_csv('data/gazeta/gazeta_train.csv')
        data_val = pd.read_csv('data/gazeta/gazeta_val.csv')

    if args.train_tokenizer:
        tokenizer = CustomTokenizer()
        tokenizer.train(data['text'].values, dir_path=args.tokenizer_path)
    else:
        tokenizer = CustomTokenizer.load_from_pretrained(
            os.path.join(args.tokenizer_path, 'vocab.json'), 
            os.path.join(args.tokenizer_path, 'merges.txt'), 
            args.max_length
            )

    bart_config = BartConfig(
        vocab_size = tokenizer.get_vocab_size(), 
        pad_token_id = tokenizer.token_to_id("<pad>"),
        bos_token_id= tokenizer.token_to_id("<s>"),
        eos_token_id = tokenizer.token_to_id("</s>"),
        encoder_layers = args.encoder_layers,
        decoder_layers = args.decoder_layers,
        return_dict=True
        )
    
    if args.pretraining:
        model = BART(
            bart_config = bart_config,
            parameters = args,
            data_train = data_train,
            data_val = data_val,
            tokenizer = tokenizer
            )
    else:
        model = BART_finetune(
            bart_config = bart_config,
            parameters = args,
            data_train = data_train,
            data_val = data_val,
            tokenizer = tokenizer
        )

    # trainer = Trainer(
    #     gpus=1, max_epochs=args.n_epochs, 
    #     callbacks = [early_stop_callback, checkpoint_callback], 
    #     )
    # trainer.fit(model)
