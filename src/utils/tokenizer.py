from tokenizers.processors import BertProcessing
from tokenizers import ByteLevelBPETokenizer
from pathlib import Path
import os


class CustomTokenizer(ByteLevelBPETokenizer):
    special_tokens=["<unk>","<pad>", "<mask>", "<s>", "</s>"]

    def __init__(self, vocab = None, merges = None):
        super().__init__(vocab, merges)

    def train(self, data, vocab_size = 35_000):
        if isinstance(data, str):
            super().train(
                data, 
                vocab_size = vocab_size,
                special_tokens=self.special_tokens
            )
        else:
            super().train_from_iterator(
                data, 
                vocab_size = vocab_size,
                special_tokens=self.special_tokens
            )

        dir_path = Path(__file__).parents[2]/'model/tokenizer'
        if not dir_path.exists():
            os.makedirs(dir_path)
        self.save_model(str(dir_path))

    @classmethod
    def load_from_pretrained(
        cls,
        max_length = 750
        ):
        vocab = Path(__file__).parents[2]/'model/tokenizer/vocab.json'
        merges = Path(__file__).parents[2]/'model/tokenizer/merges.txt'

        tokenizer = cls(str(vocab), str(merges))
        tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", tokenizer.token_to_id("</s>")),
            ("<s>", tokenizer.token_to_id("<s>")),
            )
        tokenizer.enable_truncation(max_length=max_length)
        return tokenizer
