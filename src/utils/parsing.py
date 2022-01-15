import asyncio
import aiohttp
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
import re
import os
import time


async def fetch(client, url):
    async with client.get(url) as resp:
        return await resp.read()

async def get_pages(n_pages):
    url = 'https://svpressa.ru/all/news-archive/'
    sema = asyncio.BoundedSemaphore(5)
    tasks = []

    async with aiohttp.ClientSession() as client:
        async with sema:   
            for i in range(1, n_pages):
                tasks.append(
                    asyncio.ensure_future(fetch(client, f'{url}?page={i}'))
                )
            tasks = await asyncio.gather(*tasks)
    return tasks

async def get_articles(page):
    soup = BeautifulSoup(page.decode('utf-8'))
    sema = asyncio.BoundedSemaphore(5)
    short_texts = []
    articles = []

    async with aiohttp.ClientSession() as client:
        async with sema:
            for article in soup.find_all('article', {'class': 'b-article b-article_item'}):
                ref = (
                    article
                    .find('div', {'class': 'b-article__container_item'})\
                    .find('a')
                )
                link = f"https://svpressa.ru{ref['href']}"

                short_texts.append(ref.text)
                articles.append(
                    asyncio.ensure_future(fetch(client, link))
                )

            articles = await asyncio.gather(*articles)
    return short_texts, articles        

async def get_data(n_pages, chunk_size):
    pages = await get_pages(n_pages)
    data_short = []
    data_full = []

    for i, page in tqdm(enumerate(pages, 1)):
        #clear RAM
        if chunk_size%1 == 0:
            df = pd.DataFrame({'data_full':data_full, 'data_short':data_short})
            if not os.path.exists('data/press'):
                os.mkdir('data/press')
            df.to_csv(f'data/press/data_{i//chunk_size}.csv', index = False)
            data_short = []
            data_full = []

        short_texts, articles = await get_articles(page)
        data_short += short_texts

        for article in articles:
            soup = BeautifulSoup(article.decode('utf-8'))
            all_texts = soup.find('div', {'class': 'b-text__content'}).find_all('p')
            article = filter(lambda x: not str(x).startswith('<p>\n<a href') and len(x.text) > 1, all_texts)
            article = ' '.join(list(map(lambda x: x.text, article)))
            article = re.sub('\xa0', ' ', article)
            data_full.append(article)
        

# start_time = time.time()
# print(asyncio.run(main(100))[:2])
# print("--- %s seconds ---" % (time.time() - start_time))
#asyncio.run(get_data(100, 50))