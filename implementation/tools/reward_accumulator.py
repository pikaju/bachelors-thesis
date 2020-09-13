import pandas as pd
import matplotlib.pyplot as plt

ax = plt.gca()

for run in ['state_lr_0.0', 'state_lr_0.5', 'state_lr_1.0']:
    frame = pd.DataFrame()
    for test in range(0, 15):
        url = 'http://localhost:6006/data/plugin/scalars/scalars?tag=reward%2Ftotal&run={}%2Ftest{}&format=csv'.format(
            run, test)
        print(url)
        data = pd.read_csv(url)
        frame = pd.concat([frame, data])
    frame = frame.groupby(frame.index).median()
    frame = frame.head(256).rename(columns={'Value': run})
    # idx = len(frame) - 1 if len(frame) % 2 else len(frame)
    # frame = frame[:idx].groupby(frame.index[:idx] // 2).mean()
    frame.plot(x='Step', y=run, ax=ax)

plt.show()
