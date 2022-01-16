from transformers import BartForConditionalGeneration
from pytorch_lightning.core.lightning import LightningModule
from nltk.translate.bleu_score import corpus_bleu
from src.loaders import PretrainLoader
import pandas as pd
import torch


class BART(LightningModule):
    def __init__(
        self, 
        bart_config: dict, 
        parameters: dict, 
        data_train: pd.DataFrame, 
        data_val: pd.DataFrame, 
        col_article: str,
        col_summary: str,
        tokenizer
        ):
        super().__init__()

        self.bart = BartForConditionalGeneration(bart_config)
        self.data_train = data_train
        self.data_val = data_val
        self.col_article = col_article
        self.col_summary = col_summary
        self.tokenizer = tokenizer
        self.cfg = parameters
        self.save_hyperparameters(ignore=["data_train", "data_val", "tokenizer"])

    def forward(self, decoder_input_ids = None, input_ids=None, labels = None, attention_mask=None):
        bart_output = self.bart(input_ids=input_ids,
                                attention_mask=attention_mask,
                                decoder_input_ids = decoder_input_ids,
                                labels = labels
                                )
        return bart_output

    def __dataloader(self, stage = 'train'):
        return PretrainLoader.load(
            data = self.data_train if stage == 'train' else self.data_val,
            tokenizer = self.tokenizer, 
            max_length = self.cfg.max_length,
            batch_size = self.cfg.batch_size,
            sentence_permutation = self.cfg.get('sentence_permutation', True),
            token_masking = self.cfg.get('token_masking', False),
            token_deletion = self.cfg.get('token_deletion', False),
            text_infilling = self.cfg.get('text_infilling', True),
            shuffle = True if stage == 'train' else False
            )

    def train_dataloader(self):
        return self.__dataloader('train')
    
    def val_dataloader(self):
        return self.__dataloader('val')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.cfg.get('step', 1), gamma = self.cfg.get('gamma', 0.7))
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, attn, y, labels = batch['input_ids'], batch['attention_mask'], batch['decoder_input_ids'], batch['lm_labels']
        output = self(decoder_input_ids = y[:, :-1], input_ids = x, attention_mask = attn, labels = labels)
        tensorboard_logs = {'train_loss': output.loss}
        return {'loss': output.loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, attn, y, labels = batch['input_ids'], batch['attention_mask'], batch['decoder_input_ids'], batch['lm_labels']
        output = self(decoder_input_ids = y[:, :-1], input_ids = x, attention_mask = attn, labels = labels)

        predictions = torch.argmax(output.logits, dim=-1)
        pred = self.tokenizer.decode_batch(predictions.cpu().numpy().tolist())

        ref = self.tokenizer.decode_batch(y.cpu().numpy().tolist())
        
        bleu = corpus_bleu([[r] for r in ref], pred) 
        bleu = torch.FloatTensor([bleu])
        return {'val_loss': output.loss, 'bleu': bleu}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_bleu = torch.stack([x['bleu'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_loss, 'avg_bleu': avg_bleu}
        self.log('avg_bleu', avg_bleu)
        return {'val_loss': avg_loss, 'avg_bleu': avg_bleu, 'progress_bar': tensorboard_logs}
