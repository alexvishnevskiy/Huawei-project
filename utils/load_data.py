from bs4 import BeautifulSoup
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import argparse
import os
import re
import json
import gzip
import shutil


def read_gazeta(path: str):
    path = Path(path)
    with open(path, 'r') as json_file:
        json_list = list(json_file)

    result = pd.DataFrame.from_dict([json.loads(json_str) for json_str in json_list])
    if not os.path.exists('./data/gazeta'):
        os.mkdir('./data/gazeta')
    result.to_csv(f'./data/gazeta/{path.stem}.csv', index = False)
    return result


class RIA:
    """
    path: path of json.gz file
    n_rows: number of rows to save
    chunk_size: chunk_size
    """
    def __init__(self, path, n_rows = 500_000, chunk_size = 10_000):
        self.path = path
        self.n_rows = n_rows
        self.chunk_size = chunk_size
        
    def _unzip_file(self):
        path = Path(self.path)
        self.json_path = os.path.join(path.parent, 'ria.json')
        with gzip.open(path, 'rb') as f_in:
            with open(self.json_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    def _preprocess_row(self, row):
        z, p = BeautifulSoup(row['text'], features="lxml"), BeautifulSoup(row['title'], features="lxml")

        text = re.sub('[\n]', ' ', '.'.join(z.text.split('.')[1:]))
        text = re.sub('[\xa0]', ' ', text)
        text = re.sub('[>>]', '', text)
        title = re.sub('[\/]','',p.text)
        return text, title

    def _preprocess_data(self, data: pd.DataFrame):
        data_processed = data.drop_duplicates('text').drop_duplicates('title')
        data_processed[['text', 'title']] = data_processed[['text', 'title']].astype('str')

        condition = (
            data_processed
            .apply(lambda x: len(x['text'].split(' ')) > 10 
                             and len(x['title'].split(' ')) > 3, axis = 1)
        )
        data_processed = data_processed[condition]
        return data_processed

    def _save_data(self, n_rows):
        with open(self.json_path, "r", encoding='utf-8') as read_file:
            data = pd.DataFrame()
            for i, row in tqdm(enumerate(read_file, 1)):
                if n_rows == i: break

                if i % self.chunk_size == 0:
                    data = self._preprocess_data(data)
                    if not os.path.exists('./data/ria'):
                        os.mkdir('./data/ria') 
                    data.to_csv(f'./data/ria/data_{i//self.chunk_size}.csv', index = False)
                    data = pd.DataFrame()

                row = json.loads(row)
                text, title = self._preprocess_row(row)
                data_dict = {'text':text, 'title':title}
                data = pd.concat([data, pd.DataFrame(data_dict, index =[i])])
        data.to_csv(f'./data/ria/data_{i//self.chunk_size}.csv', index = False)
                
    def get_data(self):
        self._unzip_file()
        self._save_data(self.n_rows)
        
    def clear_data(self):
        shutil.rmtree('./data/ria')
        os.remove(self.json_path)

def collect_data(n_rows, chunk_size):
    read_gazeta('data/gazeta_test.jsonl')
    read_gazeta('data/gazeta_train.jsonl')
    read_gazeta('data/gazeta_val.jsonl')
    RIA('data/ria.json.gz', n_rows, chunk_size).get_data()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Data collection")
    parser.add_argument("-n_rows", type = int, default=500_000)
    parser.add_argument("-chunk_size", type = int, default=10_000)
    
    args = parser.parse_args()
    collect_data(args.n_rows, args.chunk_size)
    