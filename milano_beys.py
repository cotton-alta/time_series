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
u_traffic_pred = []
traffic_pred = []

mcmc_code = """
data {
    int<lower=1> N;
    int population[N];
    int traffic[N];
}

parameters {
    real traf_mu;
    real <lower=0,upper=1> x;
}

model {
    for (i in 1:N){
        traffic[i] ~ poisson(traf_mu + x * population[i]);
    }
    x ~ normal(0, 1);
    traf_mu ~ normal(0, 1000);
}
"""

'''
mcmc_code = """
data {
    int N;
    int sms[N];
    int calls[N];
    int population[N];
    int traffic[N];
    real x[N];
}

parameters {
    real u_traffic;
    real u_sms;
    real u_calls;
    real <lower=0,upper=1> x_i;
    real <lower=0,upper=1> sigma1;
    real <lower=0,upper=1> sigma2;
    real <lower=0,upper=1> sigma3;
}

model {
    for (i in 1:N){
        sms[i] ~ normal(u_sms, sigma1);
        calls[i] ~ normal(u_calls, sigma2);
        traffic[i] ~ poisson(u_traffic * x_i * population[i]);
        u_traffic ~ normal(traffic[i] / population[i] * x_i, sigma3);
    }
}
"""
'''

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
mkdir('fit_results')

df_cdrs = pd.read_csv("./milano_population.csv")

population = df_cdrs["population"]
traffic = df_cdrs["traffic"]
# 仮置き
# x_i = df_cdrs["traffic"] * 0. + 0.8
data_num = len(traffic)
call_array = df_cdrs["calls"]
sms_array = df_cdrs["sms"]

print("----------------------------------------")
print(f"data num: {data_num}")
print("----------------------------------------")

for i in range(data_num):
    if(i > sampleNum - 1 and i < data_num):
        populationDF = population[i-(sampleNum):i:1]
        populationList = populationDF.values.tolist()
        populationInput = [int(f) for f in populationList]

        # x_iDF = x_i[i-(sampleNum):i:1]
        # x_iList = x_iDF.values.tolist()
        # x_iInput = [int(f) for f in x_iList]

        trafficDF = traffic[i-(sampleNum):i:1]
        trafficList = trafficDF.values.tolist()
        trafficInput = [int(f) for f in trafficList]

        callsDF = call_array[i-(sampleNum):i:1]
        callsList = callsDF.values.tolist()
        callsInput = [int(f) for f in callsList]

        smsDF = sms_array[i-(sampleNum):i:1]
        smsList = smsDF.values.tolist()
        smsInput = [int(f) for f in smsList]

        standata = {
            'N': len(trafficInput),
            'traffic': trafficInput,
            'population': populationInput
        }
        print(standata)

        sm = pystan.StanModel(model_code=mcmc_code)
        fit_nuts = sm.sampling(data=standata, chains=5, iter=5000)

        print(fit_nuts)

        ms = fit_nuts.extract()
        traffic_pred.append(np.mean(ms['traf_mu']))

        print('-----------------------------')
        print(f'i: {i}')
        print('-----------------------------')

        plt.figure()
        fit_nuts.plot()
        plt.savefig(f'fit_results/fit_result_{i}.png')
        plt.close()
        # x_pred = np.mean(ms['x_i'])
        # print(traffic)
        # print(traffic[i])
        # traffic_pred.append(traffic[i])

df = pd.DataFrame(traffic_pred, columns=['predicted-traffic'])
# df['u_traffic'] = u_traffic_pred

df.to_csv('milano_beys_result.csv')
