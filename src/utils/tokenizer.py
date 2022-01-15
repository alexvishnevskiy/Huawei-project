from tokenizers.processors import BertProcessing
from tokenizers import ByteLevelBPETokenizer
import os


class CustomTokenizer(ByteLevelBPETokenizer):
    special_tokens=["<unk>","<pad>", "<mask>", "<s>", "</s>"]

    def __init__(self, vocab = None, merges = None):
        super().__init__(vocab, merges)

    def train(self, data, vocab_size = 35_000, dir_path = 'model/tokenizer'):
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

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self.save_model(dir_path)

    @classmethod
    def load_from_pretrained(
        cls, 
        vocab = 'model/tokenizer/vocab.json', 
        merges = 'model/tokenizer/merges.txt', 
        max_length = 750
        ):

        tokenizer = cls(vocab, merges)
        tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", tokenizer.token_to_id("</s>")),
            ("<s>", tokenizer.token_to_id("<s>")),
            )
        tokenizer.enable_truncation(max_length=max_length)
        return tokenizer
