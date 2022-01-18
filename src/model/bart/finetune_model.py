from distutils.command.config import config
from src.utils.tokenizer import CustomTokenizer
from transformers import BartForConditionalGeneration
from pytorch_lightning.core.lightning import LightningModule
from nltk.translate.bleu_score import corpus_bleu
from src.loaders import FineTuneLoader
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch


class BART_finetune(LightningModule):
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

        self.bart = self.__load_model(bart_config, parameters.get('pretrained_path'))
        self.tokenizer = tokenizer
        self.data_train = data_train
        self.data_val = data_val
        self.col_article = col_article
        self.col_summary = col_summary
        self.cfg = parameters

    def forward(self, decoder_input_ids = None, input_ids=None, attention_mask=None, labels = None):
        bart_output = self.bart(input_ids=input_ids,
                                attention_mask=attention_mask,
                                decoder_input_ids = decoder_input_ids,
                                labels = labels)
        return bart_output

    def __load_model(self, bart_config, pretrained_path = None):
        if pretrained_path is not None:
            bart = load_bart(bart_config, pretrained_path)
        else:
            bart = BartForConditionalGeneration(bart_config)
        return bart

    def __dataloader(self, stage = 'train'):
        return FineTuneLoader.load(
            data = self.data_train if stage == 'train' else self.data_val,
            tokenizer = self.tokenizer, 
            col_article = self.col_article,
            col_summary = self.col_summary,
            batch_size = self.cfg.batch_size,
            max_length = self.cfg.max_length,
            shuffle = True if stage == 'train' else False
            )

    def train_dataloader(self):
        return self.__dataloader('train')
    
    def val_dataloader(self):
        return self.__dataloader('val')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr = self.cfg.max_lr,
            pct_start = self.cfg.pct_start,
            total_steps = int(
                len(self.train_dataloader())*
                self.cfg.num_epoch/self.cfg.acc_step
                )
            )
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': self.cfg.acc_step
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        output = self(**batch)
        tensorboard_logs = {'train_loss': output.loss}
        return {'loss': output.loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        output = self(**batch)

        predictions = torch.argmax(output.logits, dim=-1)
        pred = predictions.cpu().numpy()
        pred = [l[l != self.tokenizer.token_to_id("<pad>")].tolist() for l in pred]
        pred = self.tokenizer.decode_batch(pred)

        labels = batch['labels'].cpu().numpy()
        labels = [l[l != -100].tolist() for l in labels]
        ref = self.tokenizer.decode_batch(labels)
        
        bleu = corpus_bleu([[r] for r in ref], pred) 
        bleu = torch.FloatTensor([bleu])
        return {'val_loss': output.loss, 'bleu': bleu}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_bleu = torch.stack([x['bleu'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_loss, 'bleu': avg_bleu}
        self.log('val_loss', avg_loss)
        self.log('bleu', avg_bleu)
        return {'val_loss': avg_loss, 'bleu': avg_bleu, 'progress_bar': tensorboard_logs}

def load_bart(config, path):
    bart = BartForConditionalGeneration(config)
    old_state_dict = torch.load(path)['state_dict']
    new_state_dict = OrderedDict()
    
    for k, v in old_state_dict.items():
        new_state_dict[k[5:]] = v

    bart.load_state_dict(new_state_dict)
    return bart

@torch.no_grad()
def generate_summary(config, path, text, max_length = 750, num_beams = 5):
    model = load_bart(config, path)
    model.eval()
    model.cuda()

    tokenizer = CustomTokenizer.load_from_pretrained(max_length)
    token_to_id = lambda x: tokenizer.token_to_id(x)
    special_tokens = [token_to_id(x) for x in ["<pad>", "<s>", "</s>", "<mask>"]]

    ids = np.asarray(tokenizer.encode(text).ids)
    input_ids = torch.from_numpy(ids)

    output = model.generate(
        inputs = input_ids.unsqueeze(0).cuda(),
        max_length = max_length,
        num_beams = num_beams
        )
    output = output.squeeze().cpu().numpy().tolist()
    output = list(
        filter(lambda x: all([x != spec_t for spec_t in special_tokens]), output)
    )
    output = tokenizer.decode(output)
    return output
