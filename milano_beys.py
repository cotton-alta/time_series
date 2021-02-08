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

mcmc_code = """
data {
    int<lower=1> N;
    int population[N];
    int traffic[N];
}

parameters {
    real per_person;
    real <lower=0,upper=1> x;
}

model {
    for (i in 1:N){
        traffic[i] ~ poisson(x * population[i] * per_person);
    }
    x ~ normal(0, 1);
    per_person ~ normal(0, 100);
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
mkdir('fit_results')

df_cdrs = pd.read_csv("./milano_population.csv")

population = df_cdrs["population"]
traffic = df_cdrs["traffic"]
data_num = len(traffic)
# call_array = df_cdrs["calls"]
# sms_array = df_cdrs["sms"]
sampleNum = 5
x_pred = []
per_person_pred = []

print("----------------------------------------")
print(f"data num: {data_num}")
print("----------------------------------------")

for i in range(data_num):
    if(i > sampleNum - 1 and i < data_num):
        populationDF = population[i-(sampleNum):i:1]
        populationList = populationDF.values.tolist()
        populationInput = [int(f) for f in populationList]

        trafficDF = traffic[i-(sampleNum):i:1]
        trafficList = trafficDF.values.tolist()
        trafficInput = [int(f) for f in trafficList]

        # callsDF = call_array[i-(sampleNum):i:1]
        # callsList = callsDF.values.tolist()
        # callsInput = [int(f) for f in callsList]

        # smsDF = sms_array[i-(sampleNum):i:1]
        # smsList = smsDF.values.tolist()
        # smsInput = [int(f) for f in smsList]

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

        plt.figure()
        fit_nuts.plot()
        plt.savefig(f'fit_results/fit_result_{i}.png')
        plt.close()
        x = np.mean(ms['x'])
        per_person = np.mean(ms['per_person'])

        per_person_pred.append(per_person)
        x_pred.append(x)

df = pd.DataFrame({ 'predicted_x': x_pred, 'predicted_per_person': per_person_pred })

df.to_csv('milano_beys_result.csv')
