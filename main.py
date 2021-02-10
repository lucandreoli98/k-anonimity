import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


def import_csv_dataset():
    import_fields = pd.read_csv('redacted-2020-june-30-wprdc-.csv', header=None).to_numpy()[0, :]
    import_values = pd.read_csv('redacted-2020-june-30-wprdc-.csv').to_numpy()
    return import_fields, import_values


def import_csv_to_generalize():
    import_gen_file = pd.read_csv('redacted-2020-june-30-wprdc-.csv').to_numpy()
    return


def remove_ei(remove_fields: np.ndarray, remove_values: np.ndarray):
    remove_fields = remove_fields[2:10]
    remove_values = remove_values[:, 2:10]
    return remove_fields, remove_values


def data2int(data_values: np.ndarray, idx=None):
    if idx is None:
        idx = []
    for j in idx:
        for k in range(data_values.shape[0]):
            if type(data_values[k, j]) is not float:
                data_values[k, j] = int(str(data_values[k, j]).replace('-', ''))
    return data_values


def generalize_data(values_to_gen: np.ndarray, qi_data_idx_to_gen: int):
    for j in range(values_to_gen.shape[0]):
        if type(values_to_gen[j, qi_data_idx_to_gen]) is not float:
            if values_to_gen[j, qi_data_idx_to_gen] in range(10000, 100000000):
                values_to_gen[j, qi_data_idx_to_gen] = int(np.trunc(values_to_gen[j, qi_data_idx_to_gen] / 100))
            elif values_to_gen[j, qi_data_idx_to_gen] in range(10000):
                values_to_gen[j, qi_data_idx_to_gen] = np.nan
    return values_to_gen


if __name__ == '__main__':
    [fields, values] = import_csv_dataset()
    [fields, values] = remove_ei(fields, values)

    records_number = values.shape[0]
    fields_number = fields.shape[0]
    qi_idx = [0, 1, 2, 3, 4]
    print(fields)
    values = data2int(values, qi_idx[2:5])

    for i in qi_idx:
        print(list(np.unique(values[:, i])))
