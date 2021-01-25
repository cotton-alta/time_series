import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, default="./dataset") # datasetのディレクトリ
parser.add_argument("-o", "--output", type=str, default="output") # 結果出力先ディレクトリ
parser.add_argument("-t", "--technique", type=str, default="sarima") # 分析手法の選択
parser.add_argument("-c", "--cell", type=int, default=3200) # CellIDの指定
parser.add_argument("-m", "--month", type=str, default="november") # 月の選択
parser.add_argument("-s", "--startdate", type=int, default=2) # 開始日の選択
parser.add_argument("-e", "--enddate", type=int, default=14) # 終了日の選択
parser.add_argument("-u", "--hours", type=str, default="00") # 時間の選択
parser.add_argument("-p", "--person", type=int, default=10) # 1人の1時間当たりの通信量
args = parser.parse_args()
output = args.output
dataset_dir = args.dataset
technique = args.technique
cell = args.cell
month = args.month
start_date = args.startdate
end_date = args.enddate
hour = args.hours
internet_per_person = args.person

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

df_cdrs_internet = df_cdrs[["CellID", "datetime", "internet", "calls", "sms"]] \
                    .groupby(["CellID", "datetime"], as_index=False) \
                    .sum()

print("----------------------------------------")
print(df_cdrs)
print("----------------------------------------")

df_cdrs_internet["hour"] = df_cdrs_internet.datetime.dt.hour + 24 * (df_cdrs_internet.datetime.dt.day - 1)

# internet = df_cdrs["internet"]
# data_num = len(internet)
# call_array = df_cdrs["calls"]
# sms_array = df_cdrs["sms"]

df_cdrs_internet["population"] = df_cdrs_internet[df_cdrs_internet.CellID==cell]["internet"] / internet_per_person
df_cdrs_internet["population"] = df_cdrs_internet["population"].ewm(span=10).mean()
df_cdrs_internet["traffic"] = df_cdrs_internet["population"] * internet_per_person
f = plt.figure()

df_cdrs_internet = df_cdrs_internet[df_cdrs_internet.CellID==cell].drop_duplicates(subset="hour")
df_cdrs_internet = df_cdrs_internet.set_index(["hour"]).sort_index()

ax = df_cdrs_internet[df_cdrs_internet.CellID==cell]["population"].plot(label="population")
sns.despine()

box = ax.get_position()

ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=5)

plt.legend()

plt.xlabel("weekly hours")
plt.ylabel("number of connections")
plt.savefig("population.png")

df_cdrs_internet.to_csv('milano_population.csv')
print(df_cdrs)
