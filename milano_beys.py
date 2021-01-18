import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pystan
from scipy.stats import norm
import math
from itertools import zip_longest
import datetime
import time
import argparse
import os

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15,10

sampleNum = 5
u_internet_pred = []
internet_pred = []
u_calls = []
u_sms = []

mcmc_code = """
data {
    int N;
    int sms[N];
    int calls[N];
    int internet[N];
}

parameters {
    real u_internet;
    real u_sms;
    real u_calls;
    real <lower=0,upper=1> sigma1;
    real <lower=0,upper=1> sigma2;
    real <lower=0,upper=1> sigma3;
}

model {
    for (i in 1:N){
        sms[i] ~ normal(u_sms, sigma1);
        calls[i] ~ normal(u_calls, sigma2);
        internet[i] ~ poisson(u_internet);
        u_internet ~ normal(internet[i], sigma3);
    }
}
"""

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, default="./dataset") # datasetのディレクトリ
parser.add_argument("-o", "--output", type=str, default="output") # 結果出力先ディレクトリ
parser.add_argument("-t", "--technique", type=str, default="sarima") # 分析手法の選択
parser.add_argument("-c", "--cell", type=int, default=3200) # CellIDの指定
parser.add_argument("-m", "--month", type=str, default="november") # 月の選択
parser.add_argument("-s", "--startdate", type=int, default=2) # 開始日の選択
parser.add_argument("-e", "--enddate", type=int, default=8) # 終了日の選択
parser.add_argument("-u", "--hours", type=str, default="00") # 時間の選択
args = parser.parse_args()
output = args.output
dataset_dir = args.dataset
technique = args.technique
cell = args.cell
month = args.month
start_date = args.startdate
end_date = args.enddate
hour = args.hours

def mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

mkdir(output)

if month == "november":
    dates = ["2013-11-{0:02}".format(n) for n in range(start_date, end_date + 1)]
    directory = [f"{dataset_dir}/milano/full-November/sms-call-internet-mi-{date}.txt" for date in dates]
else:
    dates = ["2013-12-{0:02}".format(n) for n in range(start_date, end_date + 1)]
    directory = [f"{dataset_dir}/milano/full-December/sms-call-internet-mi-{date}.txt" for date in dates]

print("----------------------------------------")
print(f"month: {month}")
print("----------------------------------------")

df_cdrs = pd.DataFrame({})

for file in directory:
    df = pd.read_csv(
        file,
        names=("CellID", "datetime", "countrycode", "smsin", "smsout", "callin", "callout", "internet"),
        delimiter="\t",
        parse_dates=["datetime"]
    )
    df_cdrs = df_cdrs.append(df)

df_cdrs = df_cdrs.fillna(0)
df_cdrs["datetime"] = pd.to_datetime(df_cdrs["datetime"], unit="ms")
df_cdrs["sms"] = df_cdrs["smsin"] + df_cdrs["smsout"]
df_cdrs["calls"] = df_cdrs["callin"] + df_cdrs["callout"]

print("----------------------------------------")
print(df_cdrs)
print("----------------------------------------")

df_cdrs["hour"] = df_cdrs.datetime.dt.hour + 24 * (df_cdrs.datetime.dt.day - 1)

df_cdrs = df_cdrs[df_cdrs.CellID==cell].drop_duplicates(subset="hour")
df_cdrs = df_cdrs.reset_index()

print(df_cdrs)

internet = df_cdrs["internet"]
data_num = len(internet)
call_array = df_cdrs["calls"]
sms_array = df_cdrs["sms"]

print("----------------------------------------")
print(f"data num: {data_num}")
print("----------------------------------------")

for i in range(data_num):
    if(i > sampleNum - 1 and i < data_num):
        internetDF = internet[i-(sampleNum):i:1]
        internetList = internetDF.values.tolist()
        internetInput = [int(f) for f in internetList]

        callsDF = call_array[i-(sampleNum):i:1]
        callsList = callsDF.values.tolist()
        callsInput = [int(f) for f in callsList]

        smsDF = sms_array[i-(sampleNum):i:1]
        smsList = smsDF.values.tolist()
        smsInput = [int(f) for f in smsList]

        standata = {
            'N': len(internetInput),
            'calls': callsInput,
            'sms': smsInput,
            'internet': internetInput
        }
        print(standata)

        sm = pystan.StanModel(model_code=mcmc_code)
        fit_nuts = sm.sampling(data=standata, chains=5, iter=5000)

        print(fit_nuts)

        ms = fit_nuts.extract()
        u_internet_pred.append(np.mean(ms['u_internet']))
        print(internet)
        print(internet[i])
        internet_pred.append(internet[i])

df = pd.DataFrame(internet_pred, columns=['predicted-internet'])
df['u_internet'] = u_internet_pred
df.to_csv('milano_beys_result.csv')
