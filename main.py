import math
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


def import_csv():
    import_fields = pd.read_csv('redacted-2020-june-30-wprdc-.csv', header=None).to_numpy()[0, :]
    import_values = pd.read_csv('redacted-2020-june-30-wprdc-.csv').to_numpy()
    return import_fields, import_values


def remove_ei(remove_fields: np.ndarray, remove_values: np.ndarray):
    remove_fields = remove_fields[2:10]
    remove_values = remove_values[:, 2:10]
    return remove_fields, remove_values


def data2int(data_values: np.ndarray, idx=None):
    if idx is None:
        idx = []
    for j in idx:
        for i in range(data_values.shape[0]):
            if (type(data_values[i, j])) is not float:
                data_values[i, j] = int(str(data_values[i, j]).replace('-', ''))


if __name__ == '__main__':
    [fields, values] = import_csv()
    [fields, values] = remove_ei(fields, values)

    records_number = values.shape[0]
    print(fields)
    qi_idx = [0, 1, 2, 3, 4]
    data_idx = [2, 3, 4]
    data2int(values, data_idx)

