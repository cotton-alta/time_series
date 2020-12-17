import os
import argparse
import csv
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--dataset", type=str, default="./dataset") # datasetのディレクトリ
parser.add_argument("-o", "--output", type=str, default="output") # 結果出力先ディレクトリ
parser.add_argument("-t", "--technique", type=str, default="sarima") # 分析手法の選択
parser.add_argument("-c", "--countrycode", type=str, default="0") # 分析手法の選択
parser.add_argument("-m", "--month", type=str, default="november") # 月の選択
parser.add_argument("-d", "--date", type=str, default="01") # 日の選択
parser.add_argument("-u", "--hours", type=str, default="00") # 時間の選択
args = parser.parse_args()
output = args.output
dataset_dir = args.dataset
technique = args.technique
countrycode = args.countrycode
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

df_cdrs = pd.DataFrame({})

for file in directory:
    df = pd.read_csv(
        file,
        names=("CellID", "datetime", "countrycode", "smsin", "smsout", "callin", "callout", "internet"),
        delimiter="\t",
        parse_dates=["datetime"]
    )
    df_cdrs = df_cdrs.append(df)
    break

df_cdrs["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
df_cdrs = df_cdrs.fillna(0)
df_cdrs["sms"] = df_cdrs["smsin"] + df_cdrs["smsout"]
df_cdrs["calls"] = df_cdrs["callin"] + df_cdrs["callout"]

df_cdrs_internet = df_cdrs[["CellID", "datetime", "internet", "calls", "sms"]] \
                    .groupby(["CellID", "datetime"], as_index=False) \
                    .sum()

df_cdrs_internet["hour"] = df_cdrs_internet.datetime.dt.hour + 24 * (df_cdrs_internet.datetime.dt.day - 1)
df_cdrs_internet = df_cdrs_internet.set_index(["hour"]).sort_index()

print(df_cdrs_internet)

f = plt.figure()

ax = df_cdrs_internet[df_cdrs_internet.CellID==3200]["internet"].plot(label="human001")

plt.xlabel("weekly hour")
plt.ylabel("number of connections")
sns.despine()

box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=5)


# 仮パラメータを配置
p = 1
d = 1
q = 3
sp = 0
sd = 1
sq = 1

sarima = sm.tsa.SARIMAX(
            df_cdrs_internet[df_cdrs_internet.CellID==3200]["internet"],
            order=(p, d, q),
            seasonal_order=(sp, sd, sq, 4),
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit()

print(sarima.summary())

# ts_pred = sarima.predict(start="2013-11-13 22:50:00", end="2013-11-13 23:30:00")
ts_pred = sarima.predict(start=300, end=320)

plt.plot(ts_pred, label="future", color="red")
plt.savefig("sample.png")

print(ts_pred)
