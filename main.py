import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from collections import Counter


def import_csv_dataset():
    """
    Imports csv file for read data and the header fields, it cleans the string values

    :return: fields of dataset and data to be anonymized
    """
    import_fields = pd.read_csv('redacted-2020-june-30-wprdc-.csv', header=None).to_numpy()[0, :]
    import_values = pd.read_csv('redacted-2020-june-30-wprdc-.csv').to_numpy()
    import_values = clean_values(import_values)
    return import_fields, import_values


def clean_values(values_to_clean: np.ndarray):
    """
    Cleans string values of dataset to improve string operations and comparison without leads damage to the meaning

    :param values_to_clean: Entire dataset including 'Last Name' and 'First Name' fields
    :return: Cleaned string values ('Job title' [:,2], 'Department' [:,3]) from char_rem specified special chars
    """
    char_rem = "!@#$%^*()[]{};:.,/<>?|`~-=_+'\\"
    for j in range(values_to_clean.shape[0]):
        for k in range(2, 4):
            for c in char_rem:
                values_to_clean[j, k] = re.sub(' +', ' ', values_to_clean[j, k].replace(c, " ").strip())
    return values_to_clean


def remove_ei(remove_fields: np.ndarray, remove_values: np.ndarray):
    """
    Removes the Explicit Identifiers on dataset to be anonymized

    :param remove_fields: Entire dataset fields
    :param remove_values: Entire dataset values
    :return: the fields and values without the EI
    """
    remove_fields = remove_fields[2:10]
    remove_values = remove_values[:, 2:10]
    return remove_fields, remove_values


def remove_unuseful(remove_fields: np.ndarray, remove_values: np.ndarray):
    """
    Removes the unuseful Sensitive Data on dataset to be anonymized

    :param remove_fields: Entire dataset fields
    :param remove_values: Entire dataset values
    :return: the fields and values without unuseful SD
    """
    remove_fields = remove_fields[[0, 1, 2, 3, 4, 6]]
    remove_values = remove_values[:, [0, 1, 2, 3, 4, 6]]
    return remove_fields, remove_values


def data2int(data_values: np.ndarray, idx=None):
    """
    Convert the data values in integer like the following example: 1998-05-27 -> 19980527

    :param data_values: Entire dataset
    :param idx: Array of indices of data fields of the dataset
    :return: Entire dataset with data values converted
    """
    if idx is None:
        idx = []
    for j in idx:
        for k in range(data_values.shape[0]):
            if type(data_values[k, j]) is not float:
                data_values[k, j] = int(str(data_values[k, j]).replace('-', ''))
    return data_values


def create_string_generalize_hierarchy(values_to_get_generalize: np.ndarray, idx: int):
    """
    Creates the generalization hierarchy for 'Job Title' or 'Department' field supposing that the first word of the
    value generalize it

    :param values_to_get_generalize: Entire dataset
    :param idx: Index of 'Job Title' or 'Department' field
    :return: A matrix composed by rows, one for each domain value (of the field specified in the idx param), with the
        following structure: ['Not generalized','1st generalization level','Last generalization level']
    """
    gen_file = []
    import_fixed_values = pd.read_csv('fix_jobs.csv', header=None).to_numpy()
    values_generalized = np.unique(values_to_get_generalize[:, idx])
    for j in range(values_generalized.shape[0]):
        if len(values_generalized[j].split(" ")[0]) < 4 and len(values_generalized[j].split(" ")) > 1:
            first_level_gen = values_generalized[j].split(" ")[0] + " " + values_generalized[j].split(" ")[1]
        else:
            first_level_gen = values_generalized[j].split(" ")[0]

        if first_level_gen in import_fixed_values[:, 0] and idx == 0:
            first_level_gen = import_fixed_values[np.where(first_level_gen == import_fixed_values[:, 0]), 1]

        tmp = np.append(np.append(values_generalized[j], first_level_gen), '*')

        if j == 0:
            gen_file = tmp
        else:
            gen_file = np.concatenate((gen_file, tmp))

    gen_file = np.reshape(gen_file, (values_generalized.shape[0], 3))
    return gen_file


def generalize_data(values_to_gen: np.ndarray, qi_data_idx_to_gen: int):
    """
    Generalize the data values in the following way:    (19980527)
    the first time is called it simply delete the day   (199805)
    the second time it also remove the month            (1998)
    the third time it put 'nan' in the data values      (nan)

    :param values_to_gen: Entire dataset
    :param qi_data_idx_to_gen: Index of the data field to be generalized
    :return: Entire dataset with data values generalized
    """
    if qi_data_idx_to_gen in range(2, 5):
        for j in range(values_to_gen.shape[0]):
            if type(values_to_gen[j, qi_data_idx_to_gen]) is not float:
                if values_to_gen[j, qi_data_idx_to_gen] in range(10000, 100000000):
                    values_to_gen[j, qi_data_idx_to_gen] = int(np.trunc(values_to_gen[j, qi_data_idx_to_gen] / 100))
                elif values_to_gen[j, qi_data_idx_to_gen] in range(10000):
                    values_to_gen[j, qi_data_idx_to_gen] = np.nan
    return values_to_gen


def string_generalize(values_to_gen: np.ndarray, qi_string_idx_to_gen: int, level_of_generalization: int):
    """
    Generalizes the string values: 'Job Title' or 'Department' with specified level of generalization
    by appropriate param

    :param values_to_gen: Entire dataset
    :param qi_string_idx_to_gen: 'Job Title' (qi_string_idx_to_gen = 0) or 'Department' (qi_string_idx_to_gen = 1)
    :param level_of_generalization: Specify level of generalization (1 or 2)
    :return: Entire dataset with generalized values
    """
    if qi_string_idx_to_gen in range(0, 2):
        if level_of_generalization in range(1, 3):
            hierarchy = []
            if qi_string_idx_to_gen == 0:
                hierarchy = create_string_generalize_hierarchy(values, qi_string_idx_to_gen)
            elif qi_string_idx_to_gen == 1:
                hierarchy = create_string_generalize_hierarchy(values, qi_string_idx_to_gen)

            for j in range(values_to_gen.shape[0]):

                if len(values_to_gen[j, qi_string_idx_to_gen].split()[0]) < 4 \
                        and len(values_to_gen[j, qi_string_idx_to_gen].split()) > 1:
                    values_to_gen[j, qi_string_idx_to_gen] \
                        = values_to_gen[j, qi_string_idx_to_gen].split()[0] + " " + values_to_gen[j, qi_string_idx_to_gen].split()[1]
                else:
                    values_to_gen[j, qi_string_idx_to_gen] = values_to_gen[j, qi_string_idx_to_gen].split()[0]

            for j in range(hierarchy.shape[0]):
                indices = np.where(values_to_gen[:, qi_string_idx_to_gen] == hierarchy[j, level_of_generalization - 1])
                values_to_gen[indices, qi_string_idx_to_gen] = hierarchy[j, level_of_generalization]

    return values_to_gen


def check_job_title_occ(data: np.ndarray):
    [arr, count] = np.unique(data[:, 0], return_counts=True)
    print(list(arr[np.where(count < 4)]))
    print((arr[np.where(count < 4)].shape[0]))
    print(list(zip(list(arr[np.where(count > 10)]), list(count[np.where(count > 10)]))))
    # print(list(count[np.where(count > 10)]))


def plot_graphs(data: np.ndarray, labels: np.ndarray, idx: int):
    """
    Plot histogram of frequencies of values in the dataset to have an overview of the distribution of them
    In case of data type the x label is from 1965 to 2025
    In case of earnings type the x label is free interpretation of plot function that cover all values

    :param data: Entire dataset
    :param labels: Fields of dataset for write the title of plots
    :param idx: Index of field of the dataset considered for plot the data frequency (0-7)
    """
    if idx in range(0, 2):
        plt.figure()
        plt.hist(data[:, idx], bins=np.unique(data[:, idx]).shape[0])
        plt.title(labels[idx] + " distribution")
        plt.ylim([0, 10])
        plt.show()
    elif idx in range(2, 5):
        data_of_data = []
        for j in range(data.shape[0]):
            if type(data[j, idx]) is not float:
                if j == 0:
                    data_of_data = data[j, idx]
                else:
                    data_of_data = np.append(data_of_data, data[j, idx])

        plt.figure()
        plt.hist(np.uint16(data_of_data / 10000), bins=np.unique(np.uint16(data_of_data / 10000)).shape[0])
        plt.title(labels[idx] + " distribution")
        plt.ylim([0, 10])
        plt.xlim([1965, 2025])
        plt.grid()
        plt.show()
    elif idx == 5:
        plt.figure()
        plt.hist(data[:, idx], bins=250)
        plt.title(labels[idx] + " distribution")
        plt.ylim([0, 10])
        plt.show()


def check_k_anonymity(data: np.ndarray, k: int, qi_indices=None):
    """
    Checks if the dataset respect k-anonymity property

    :param data: Entire dataset
    :param k: k-anonymity level
    :param qi_indices: Indices of Quasi-Identifier
    :return: True if dataset respect the k-anonymity, or False
    """
    print(Counter(str(e) for e in data[:, qi_indices]))

    occurrences = list(Counter(str(e) for e in data[:, qi_indices]).values())
    for j in range(len(occurrences)):
        if occurrences[j] < k:
            return False
    return True


if __name__ == '__main__':
    # Quasi-identifier indices
    qi_idx = [0, 1, 2, 3, 4]

    # Import dataset and remove EI
    [fields, values] = import_csv_dataset()
    [fields, values] = remove_ei(fields, values)
    [fields, values] = remove_unuseful(fields, values)

    # Convert data values in integer
    values = data2int(values, qi_idx[2:5])

    # Check
    print(fields)

    values = string_generalize(values, 0, 1)

    print(list(np.unique(values[:, 0])))
