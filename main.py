import numpy as np
import pandas as pd

if __name__ == '__main__':
    fields = pd.read_csv('redacted-2020-june-30-wprdc-.csv', sep=',',).to_numpy()[0, :]
    values = pd.read_csv('redacted-2020-june-30-wprdc-.csv', sep=',', header=None).to_numpy()
    print(fields[0])
