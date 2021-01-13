import os
import argparse
import csv
import glob
import codecs
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

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
    directory = glob.glob(f"{dataset_dir}/milano/full-December/*txt")

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

df_cdrs_internet = df_cdrs[["CellID", "datetime", "internet", "calls", "sms"]] \
                    .groupby(["CellID", "datetime"], as_index=False) \
                    .sum()

# df_cdrs_internet["hour"] = (df_cdrs_internet.datetime.dt.minute / 60) + df_cdrs_internet.datetime.dt.hour + 24 * (df_cdrs_internet.datetime.dt.day - 1)
df_cdrs_internet["hour"] = df_cdrs_internet.datetime.dt.hour + 24 * (df_cdrs_internet.datetime.dt.day - 1)

# ---------------------------------------------
if month == "november":
    dates = ["2013-11-{0:02}".format(n) for n in range(start_date + 7, end_date + 8)]

    directory = [f"{dataset_dir}/milano/full-November/sms-call-internet-mi-{date}.txt" for date in dates]
else:
    directory = glob.glob(f"{dataset_dir}/milano/full-December/*txt")

df_cdrs_real = pd.DataFrame({})

for file in directory:
    df = pd.read_csv(
        file,
        names=("CellID", "datetime", "countrycode", "smsin", "smsout", "callin", "callout", "internet"),
        delimiter="\t",
        parse_dates=["datetime"]
    )
    df_cdrs_real = df_cdrs_real.append(df)

df_cdrs_real = df_cdrs_real.fillna(0)
df_cdrs_real["datetime"] = pd.to_datetime(df_cdrs_real["datetime"], unit="ms")
df_cdrs_real["sms"] = df_cdrs_real["smsin"] + df_cdrs_real["smsout"]
df_cdrs_real["calls"] = df_cdrs_real["callin"] + df_cdrs_real["callout"]

df_cdrs_internet_real = df_cdrs_real[["CellID", "datetime", "internet", "calls", "sms"]] \
                    .groupby(["CellID", "datetime"], as_index=False) \
                    .sum()

# df_cdrs_internet_real["hour"] = (df_cdrs_internet_real.datetime.dt.minute / 60) + df_cdrs_internet_real.datetime.dt.hour + 24 * (df_cdrs_internet_real.datetime.dt.day - 1)
df_cdrs_internet_real["hour"] = df_cdrs_internet_real.datetime.dt.hour + 24 * (df_cdrs_internet_real.datetime.dt.day - 1)
# ---------------------------------------------

f = plt.figure()

df_cdrs_internet_real = df_cdrs_internet_real[df_cdrs_internet_real.CellID==cell].drop_duplicates(subset="hour")
df_cdrs_internet_real = df_cdrs_internet_real.set_index(["hour"]).sort_index()

ax_real = df_cdrs_internet_real[df_cdrs_internet_real.CellID==cell]["internet"].plot(label="human001")
sns.despine()

box = ax_real.get_position()

df_cdrs_internet = df_cdrs_internet[df_cdrs_internet.CellID==cell].drop_duplicates(subset="hour")
df_cdrs_internet = df_cdrs_internet.set_index(["hour"]).sort_index()

ax = df_cdrs_internet[df_cdrs_internet.CellID==cell]["internet"].plot(label="human001")

plt.xlabel("weekly hour")
plt.ylabel("number of connections")
sns.despine()

box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=5)

# 仮パラメータを配置
p = 2
d = 1
q = 2
sp = 1
sd = 1
sq = 1

# p, qを推定する際に表示
# res = sm.tsa.arma_order_select_ic(
#             df_cdrs_internet[df_cdrs_internet.CellID==cell]["internet"],
#             ic="aic",
#             trend="nc"
# )
# print(res)

sarima = sm.tsa.SARIMAX(
            df_cdrs_internet[df_cdrs_internet.CellID==cell]["internet"],
            order=(p, d, q),
            seasonal_order=(sp, sd, sq, 61),
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit()

print(sarima.summary())

ts_pred = sarima.predict(start=180, end=400)

print("----------------------------------------")
print(ts_pred)
print("----------------------------------------")

plt.plot(ts_pred, label="future", color="red")
plt.savefig("sample.png")
