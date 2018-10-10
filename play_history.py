import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(style="whitegrid")
# plt.ion()  ## Note this correction
# fig = plt.figure()
df = pd.read_csv("reward_history.csv")
x = df.iloc[-100:, 1:].values
sns.distplot(x)
plt.show()

plt.plot(df.iloc[:, 1].values)
plt.show()