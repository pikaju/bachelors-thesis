import pandas as pd
import matplotlib.pyplot as plt

ax = plt.gca()

for run in ['lr0.5', 'lr1.0']:
    frame = pd.DataFrame()
    for test in range(0, 19):
        url = 'http://0.0.0.0:6006/data/plugin/scalars/scalars?tag=1.Total+reward%2F1.Total+reward&run={}%2Frun{}&format=csv'.format(
            run, test)
        print(url)
        data = pd.read_csv(url)
        frame = pd.concat([frame, data])
    frame = frame.groupby(frame.index).mean()
    frame = frame.rename(columns={'Value': run})
    idx = len(frame) - 1 if len(frame) % 1 else len(frame)
    frame = frame[:idx].groupby(frame.index[:idx] // 1).mean()
    frame.plot(x='Step', y=run, ax=ax)

plt.show()
