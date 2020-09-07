import pandas as pd
import matplotlib.pyplot as plt

ax = plt.gca()

for run in ['state_lr_0.0', 'state_lr_0.5', 'state_lr_1.0']:
    frame = pd.DataFrame()
    for test in range(8):
        url = 'http://localhost:6007/data/plugin/scalars/scalars?tag=reward%2Ftotal&run={}%2Ftest{}&format=csv'.format(
            run, test)
        data = pd.read_csv(url)
        frame = pd.concat([frame, data])
    frame = frame.groupby(frame.index).mean()
    frame = frame.head(80).rename(columns={'Value': run})
    idx = len(frame) - 1 if len(frame) % 2 else len(frame)
    frame = frame[:idx].groupby(frame.index[:idx] // 1).mean()
    print(frame)
    frame.plot(x='Step', y=run, ax=ax)

plt.show()
