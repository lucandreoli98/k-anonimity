import numpy as np
import pandas as pd
import re


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
    char_rem = "!@#$%^&*()[]{};:.,/<>?|`~-=_+"
    for j in range(values_to_clean.shape[0]):
        for k in range(2, 4):
            for c in char_rem:
                values_to_clean[j, k] = re.sub(' +', ' ', values_to_clean[j, k].replace(c, " ").strip())
    return values_to_clean


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
    values_generalized = np.unique(values_to_get_generalize[:, idx])
    for j in range(values_generalized.shape[0]):
        tmp = np.append(np.append(values_generalized[j], values_generalized[j].split(" ")[0]), '*')
        if j == 0:
            gen_file = tmp
        else:
            gen_file = np.concatenate((gen_file, tmp))
    gen_file = np.reshape(gen_file, (values_generalized.shape[0], 3))
    return gen_file


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


def generalize_data(values_to_gen: np.ndarray, qi_data_idx_to_gen: int):
    """
    Generalize the data values in the following way:    (19980527)
    the first time is called it simply delete the day   (199805)
    the second time it also remove the month            (1998)
    the third time it put * in the data values          (*)
    :param values_to_gen: Entire dataset
    :param qi_data_idx_to_gen: Index of the data field to be generalized
    :return: Entire dataset with data values generalized
    """
    for j in range(values_to_gen.shape[0]):
        if type(values_to_gen[j, qi_data_idx_to_gen]) is not float:
            if values_to_gen[j, qi_data_idx_to_gen] in range(10000, 100000000):
                values_to_gen[j, qi_data_idx_to_gen] = int(np.trunc(values_to_gen[j, qi_data_idx_to_gen] / 100))
            elif values_to_gen[j, qi_data_idx_to_gen] in range(10000):
                values_to_gen[j, qi_data_idx_to_gen] = np.nan
    return values_to_gen


if __name__ == '__main__':
    # Quasi-identifier indices
    qi_idx = [0, 1, 2, 3, 4]

    # Import dataset and remove EI
    [fields, values] = import_csv_dataset()
    [fields, values] = remove_ei(fields, values)

    # Convert data values in integer
    values = data2int(values, qi_idx[2:5])

    # Creation of 'Job title' and 'Department' generalization hierarchy
    job_title_hierarchy = create_string_generalize_hierarchy(values, 0)
    department_hierarchy = create_string_generalize_hierarchy(values, 1)

    # Check
    print(fields)
