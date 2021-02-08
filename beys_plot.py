import pandas as pd
import matplotlib.pyplot as plt

df_cdrs = pd.read_csv('./milano_population.csv')
df_traffic = pd.read_csv('./milano_beys_result.csv')

df = pd.DataFrame({})

df['hour'] = df_cdrs['hour']
df['traffic'] = df_cdrs['traffic']
df['predicted_x'] = df_traffic['predicted_x']
df['predicted_per_person'] = df_traffic['predicted_per_person']
df['population'] = df_cdrs['population']
df['traffic_pred'] = df['population'] * df['predicted_per_person'] * df['predicted_x']

ax_real = df['traffic'].plot(label='traffic_real')

box = ax_real.get_position()

ax_pred = df['traffic_pred'].plot(label='traffic_pred')

ax_real.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
ax_real.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=5)

plt.legend()
plt.xlabel("weekly hours")
plt.ylabel("traffic")
plt.savefig("beys_predict.png")