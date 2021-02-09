import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def import_csv():
    import_fields = pd.read_csv('redacted-2020-june-30-wprdc-.csv', header=None).to_numpy()[0, :]
    import_values = pd.read_csv('redacted-2020-june-30-wprdc-.csv').to_numpy()
    return import_fields, import_values


def remove_ei(remove_fields: np.ndarray, remove_values: np.ndarray):
    remove_fields = remove_fields[2:10]
    remove_values = remove_values[:, 2:10]
    return remove_fields, remove_values


if __name__ == '__main__':
    [fields, values] = import_csv()
    print(list(fields))
    [fields, values] = remove_ei(fields, values)
    print(list(fields))
