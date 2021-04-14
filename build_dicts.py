import os
import json
import pickle
import argparse
import re
import pandas as pd
import numpy as np

def parse_ent_list(x):
    if x.strip() == "":
        return ''

    return ' '.join([k["WikidataId"] for k in json.loads(x)])

punctuation = '!,;:?"\''
def removePunctuation(text):
    text = re.sub(r'[{}]+'.format(punctuation),'',text)
    return text.strip().lower()

parser = argparse.ArgumentParser()
parser.add_argument("--root", default="data", type=str)
parser.add_argument("--max_title", default=10, type=int)
args = parser.parse_args()

if args.root == 'data':
    print("Loading news info")
    f_train_news = os.path.join("data", "train/news.tsv")
    f_dev_news = os.path.join("data", "valid/news.tsv")
    f_test_news = os.path.join("data", "test/news.tsv")

    print("Loading training news")
    all_news = pd.read_csv(f_train_news, sep="\t", encoding="utf-8",
                            names=["newsid", "cate", "subcate", "title", "abs", "url", "title_ents", "abs_ents"],
                            quoting=3)

    print("Loading dev news")
    dev_news = pd.read_csv(f_dev_news, sep="\t", encoding="utf-8",
                            names=["newsid", "cate", "subcate", "title", "abs", "url", "title_ents", "abs_ents"],
                            quoting=3)
    all_news = pd.concat([all_news, dev_news], ignore_index=True)
    print("Loading testing news")
    test_news = pd.read_csv(f_test_news, sep="\t", encoding="utf-8",
                            names=["newsid", "cate", "subcate", "title", "abs", "url", "title_ents", "abs_ents"],
                            quoting=3)
    all_news = pd.concat([all_news, test_news], ignore_index=True)
    all_news = all_news.drop_duplicates("newsid")
else:
    all_news = pd.read_csv("{}/news.tsv".format(args.root), sep="\t", encoding="utf-8",
                            names=["newsid", "cate", "subcate", "title", "abs", "url", "title_ents", "abs_ents"],
                            quoting=3)

news_dict = {}
word_dict = {'<pad>': 0}
word_idx = 1
news_idx = 2
for n, title in all_news[['newsid', "title"]].values:
    news_dict[n] = {}
    news_dict[n]['idx'] = news_idx
    news_idx += 1

    tarr = removePunctuation(title).split()
    wid_arr = []
    for t in tarr:
        if t not in word_dict:
            word_dict[t] = word_idx
            word_idx += 1
        wid_arr.append(word_dict[t])
    cur_len = len(wid_arr)
    if cur_len < args.max_title:
        for l in range(args.max_title - cur_len):
            wid_arr.append(0)
    news_dict[n]['title'] = wid_arr[:args.max_title]

## paddning news for impression
news_dict['<pad>']= {}
news_dict['<pad>']['idx'] = 0
tarr = removePunctuation("This is the title of the padding news").split()
wid_arr = []
for t in tarr:
    if t not in word_dict:
        word_dict[t] = word_idx
        word_idx += 1
    wid_arr.append(word_dict[t])
cur_len = len(wid_arr)
if cur_len < args.max_title:
    for l in range(args.max_title - cur_len):
        wid_arr.append(0)
news_dict['<pad>']['title'] = wid_arr[:args.max_title]
## paddning news for history
news_dict['<his>']= {}
news_dict['<his>']['idx'] = 1
news_dict['<his>']['title'] = list(np.zeros(args.max_title))

print('all word', len(word_dict))
print('all news', len(news_dict))
json.dump(news_dict, open('{}/news.json'.format(args.root), 'w', encoding='utf-8'))
json.dump(word_dict, open('{}/word.json'.format(args.root), 'w', encoding='utf-8'))
