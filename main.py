import os
import argparse
import csv
import glob
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--dataset", type=str, default="./dataset") # datasetのディレクトリ
parser.add_argument("-o", "--output", type=str, default="output") # 結果出力先ディレクトリ
parser.add_argument("-t", "--technique", type=str, default="kmeans") # 分析手法の選択
parser.add_argument("-m", "--month", type=str, default="november") # 月の選択
parser.add_argument("-d", "--date", type=str, default="01") # 日の選択
parser.add_argument("-u", "--hours", type=str, default="00") # 時間の選択
args = parser.parse_args()
output = args.output
dataset_dir = args.dataset
technique = args.technique
month = args.month
date = args.date
hour = args.hours

def mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

mkdir(output)

if month == "november":
    directory = glob.glob(f"{dataset_dir}/milano/full-November/*txt")
else:
    directory = glob.glob(f"{dataset_dir}/milano/full-December/*txt")

for file in directory:
    title, _ = os.path.splitext(file)
    f = csv.reader(open(file), delimiter="\t")
    for row in f:
        print(row)
        break
