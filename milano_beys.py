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
# trafficPred = []
# utraffic1Pred = []
# utraffic2Pred = []
# uCPU_1 = []
# uCPU_2 = []
# uMEM_1 = []
# uMEM_2 = []
# xi1Pred = []
# xi2Pred = []
NPred = []
Nv1Pred = []
Nv2Pred = []

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

mcmccode = """
data {
    int N;
    int traffic_1[N];
    int traffic_2[N];
    int population[N];
    int CPU_1[N];
    int CPU_2[N];
    int MEM_1[N];
    int MEM_2[N];
}

parameters {
    real utraffic_1;
    real utraffic_2;
    real <lower=0,upper=1> xi_1;
    real <lower=0,upper=1> xi_2;
    real uCPU_1;
    real uCPU_2;
    real uMEM_1;
    real uMEM_2;
    real <lower=0,upper=1> sigma1;
    real <lower=0,upper=1> sigma2;
    real <lower=0,upper=1> sigma3;
    real <lower=0,upper=1> sigma4;
    real <lower=0,upper=1> sigma5;
    real <lower=0,upper=1> sigma6;

}

model {
    for (i in 1:N){
        CPU_1[i] ~ normal(uCPU_1*xi_1*population[i], sigma1);
        CPU_2[i] ~ normal(uCPU_2*xi_2*population[i], sigma2);
        MEM_1[i] ~ normal(uMEM_1*xi_1*population[i], sigma3);
        MEM_2[i] ~ normal(uMEM_2*xi_2*population[i], sigma4);
        traffic_1[i] ~ poisson(utraffic_1*xi_1*population[i]);
        traffic_2[i] ~ poisson(utraffic_2*xi_2*population[i]);
        utraffic_1 ~ normal(traffic_1[i]/population[i]*xi_1, sigma5);
        utraffic_2 ~ normal(traffic_2[i]/population[i]*xi_2, sigma6);
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

# df_cdrs_internet = df_cdrs[["CellID", "datetime", "internet", "calls", "sms"]] \
#                     .groupby(["CellID", "datetime"], as_index=False) \
#                     .sum()

df_cdrs["hour"] = df_cdrs.datetime.dt.hour + 24 * (df_cdrs.datetime.dt.day - 1)

cell = 3200

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

'''
for i in range(dataNum):
    if(i > sampleNum-1 and i < dataNum):
        populationDF = population[i-(sampleNum):i:1]
        populationList = populationDF.values.tolist()
        populationInput = [int(f) for f in populationList]

        traffic2DF = traffic_2[i-(sampleNum):i:1]
        traffic2List = traffic2DF.values.tolist()
        traffic2Input = [int(f) for f in traffic2List]

        CPU1DF = CPU_1[i-(sampleNum):i:1]
        CPU1List = CPU1DF.values.tolist()
        CPU1Input = [int(f) for f in CPU1List]

        MEM1DF = MEM_1[i-(sampleNum):i:1]
        MEM1List = MEM1DF.values.tolist()
        MEM1Input = [int(f) for f in MEM1List]

        sm = pystan.StanModel(model_code=mcmccode)
        fit_nuts = sm.sampling(data=standata, chains=4, iter=5000)

        print(fit_nuts)

        ms = fit_nuts.extract()
        utraffic1Pred.append(np.mean(ms['utraffic_1']))
        utraffic2Pred.append(np.mean(ms['utraffic_2']))
        predictedxi1 = np.mean(ms['xi_1'])
        predictedxi2 = np.mean(ms['xi_2'])
        xi1Pred.append(predictedxi1)
        xi2Pred.append(predictedxi2)
        NPred.append(population[i])
        Nv1Pred.append(population[i]*predictedxi1)
        Nv2Pred.append(population[i]*predictedxi2)

        #trafficPred.append(np.mean(ms['utraffic']*population[i]*predictedxi))
        trafficPred.append(np.mean(ms['utraffic_1']*population[i]*predictedxi1)+
        np.mean(ms['utraffic_2']*population[i]*predictedxi2))

df = pd.DataFrame(trafficPred,columns=['predicted-traffic'])
df['xi1'] = xi1Pred
df['xi2'] = xi2Pred
df['utraffic1'] = utraffic1Pred
df['utraffic2'] = utraffic2Pred
df['population'] = NPred
df['Nv1'] = Nv1Pred #class-1のアクティブ人口
df['Nv2'] = Nv2Pred #class-2のアクティブ人口
df.to_csv('ie1resultMix.csv')
'''
