import os
import json
import pickle
import argparse
import math
import pandas as pd
import numpy as np
import multiprocessing as mp
import random
from tqdm import tqdm

random.seed(7)

def build_examples(rank, args, df, news_info, fout):
    data_list = []
    for imp_id, hist, imp in tqdm(df[["id", "hist", "imp"]].values, total=df.shape[0]):
        if str(hist) == 'nan':
            his_list = []
        else:
            his_list = str(hist).strip().split()

        word_len = args.max_title

        his_idx_list = [news_info[h]['idx'] for h in his_list]
        his_title_list = []
        for h in his_list:
            his_title_list += news_info[h]['title']
        
        hislen = len(his_list)
        if hislen < args.max_hist_length:
            for _ in range(args.max_hist_length - hislen):
                his_idx_list.append(news_info['<his>']['idx'])
                his_title_list += news_info['<his>']['title']
        else:
            his_idx_list = his_idx_list[-args.max_hist_length:]
            his_title_list = his_title_list[-args.max_hist_length * word_len:]

        imp_list = str(imp).split(' ')
        imp_pos_list = []
        imp_neg_list = []
        for impre in imp_list:
            arr = impre.split('-')
            curn = news_info[arr[0]]['idx']
            curt = news_info[arr[0]]['title']
            label = int(arr[1])
            if label == 0:
                imp_neg_list.append((label, curt, curn))
            elif label == 1:
                imp_pos_list.append((label, curt, curn))
            else:
                raise Exception('label error!')
        
        neg_num = len(imp_neg_list)
        if neg_num < 4:
            for i in range(4 - neg_num):
                imp_neg_list.append((0, news_info['<pad>']['title'], news_info['<pad>']['idx']))
        
        for p in imp_pos_list:
            sampled = random.sample(imp_neg_list, 4)
            new_row = []
            new_row.append(int(imp_id))
            new_row.append(0)
            # idx
            new_row.append(p[2])
            for neg in sampled:
                new_row.append(neg[2])
            new_row += his_idx_list
            # title
            new_row += p[1]
            for neg in sampled:
                new_row += neg[1]
            new_row += his_title_list
            data_list.append(new_row)
    
    datanp = np.array(data_list, dtype=int)
    np.save(fout, datanp)

def main(args):
    f_train_beh = os.path.join(args.root, args.fsamples)
    df = pd.read_csv(f_train_beh, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
    news_info = json.load(open('{}/news.json'.format(args.root), 'r', encoding='utf-8'))

    subdf_len = math.ceil(len(df) / args.processes)
    cut_indices = [x * subdf_len for x in range(1, args.processes)]
    dfs = np.split(df, cut_indices)

    processes = []
    for i in range(args.processes):
        output_path = os.path.join(args.root, args.fout,  "train-{}.npy".format(i))
        p = mp.Process(target=build_examples, args=(
            i, args, dfs[i], news_info, output_path))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path options.
    parser.add_argument("--fsamples", default="train/behaviors.tsv", type=str,
                        help="Path of the training samples file.")
    parser.add_argument("--fout", default="raw", type=str,
                        help="Path of the output dir.")
    parser.add_argument("--max_hist_length", default=100, type=int,
                        help="Max length of the click history of the user.")
    parser.add_argument("--processes", default=40, type=int,
                        help="Processes number")
    parser.add_argument("--root", default="data", type=str)
    parser.add_argument("--max_title", default=10, type=int)
    args = parser.parse_args()

    main(args)

