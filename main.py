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
    import_fixed_values = None
    if idx == 0:
        import_fixed_values = pd.read_csv('fix_jobs.csv', header=None).to_numpy()
    elif idx == 1:
        import_fixed_values = pd.read_csv('fix_department.csv', header=None).to_numpy()

    values_generalized = np.unique(values_to_get_generalize[:, idx])
    for j in range(values_generalized.shape[0]):
        if len(values_generalized[j].split(" ")[0]) < 4 and len(values_generalized[j].split(" ")) > 1 and idx == 0:
            first_level_gen = values_generalized[j].split(" ")[0] + " " + values_generalized[j].split(" ")[1]
        else:
            first_level_gen = values_generalized[j].split(" ")[0]

        if idx == 1:
            first_level_gen = values_generalized[j]

        if first_level_gen in import_fixed_values[:, 0]:
            first_level_gen = import_fixed_values[np.where(first_level_gen == import_fixed_values[:, 0]), 1]

        tmp = np.append(np.append(values_generalized[j], first_level_gen), '*')

        if j == 0:
            gen_file = tmp
        else:
            gen_file = np.concatenate((gen_file, tmp))

    gen_file = np.reshape(gen_file, (values_generalized.shape[0], 3))
    return gen_file


def generalize_data(values_to_gen: np.ndarray, qi_data_idx_to_gen: int, lv: int):
    """
    Generalize the data values in the following way:    (19980527)
    the first time is called it simply delete the day   (199805)
    the second time it also remove the month            (1998)
    the third time it take the decade                   (90)
    the fourth time it put 'nan' in the data values     (nan)

    :param values_to_gen: Entire dataset
    :param qi_data_idx_to_gen: Index of the data field to be generalized
    :param lv: Level to generalize data
    :return: Entire dataset with data values generalized
    """
    tmp = []
    if qi_data_idx_to_gen in range(2, 5):
        if lv in range(1, 5):
            for j in range(values_to_gen.shape[0]):
                if type(values_to_gen[j, qi_data_idx_to_gen]) is not float:
                    if lv == 1:
                        if values_to_gen[j, qi_data_idx_to_gen] in range(1000000, 100000000):
                            if j == 0:
                                tmp = int(np.trunc(values_to_gen[j, qi_data_idx_to_gen] / 100))
                            else:
                                tmp = np.append(tmp, int(np.trunc(values_to_gen[j, qi_data_idx_to_gen] / 100)))
                    if lv == 2:
                        if values_to_gen[j, qi_data_idx_to_gen] in range(1000000, 100000000):
                            if j == 0:
                                tmp = int(np.trunc(values_to_gen[j, qi_data_idx_to_gen] / 10000))
                            else:
                                tmp = np.append(tmp, int(np.trunc(values_to_gen[j, qi_data_idx_to_gen] / 10000)))
                        elif values_to_gen[j, qi_data_idx_to_gen] in range(10000, 1000000):
                            if j == 0:
                                tmp = int(np.trunc(values_to_gen[j, qi_data_idx_to_gen] / 100))
                            else:
                                tmp = np.append(tmp, int(np.trunc(values_to_gen[j, qi_data_idx_to_gen] / 100)))
                    if lv == 3:
                        if values_to_gen[j, qi_data_idx_to_gen] in range(1000000, 100000000):
                            if j == 0:
                                tmp = int(np.trunc(
                                    (int(np.trunc(values_to_gen[j, qi_data_idx_to_gen] / 10000)) % 100) / 10) * 10)
                            else:
                                tmp = np.append(tmp, int(np.trunc(
                                    (int(np.trunc(values_to_gen[j, qi_data_idx_to_gen] / 10000)) % 100) / 10) * 10))
                        elif values_to_gen[j, qi_data_idx_to_gen] in range(10000, 1000000):
                            if j == 0:
                                tmp = int(np.trunc(
                                    (int(np.trunc(values_to_gen[j, qi_data_idx_to_gen] / 100)) % 100) / 10) * 10)
                            else:
                                tmp = np.append(tmp, int(np.trunc(
                                    (int(np.trunc(values_to_gen[j, qi_data_idx_to_gen] / 100)) % 100) / 10) * 10))
                        elif values_to_gen[j, qi_data_idx_to_gen] in range(100, 10000):
                            if j == 0:
                                tmp = int(np.trunc((values_to_gen[j, qi_data_idx_to_gen] % 100) / 10) * 10)
                            else:
                                tmp = np.append(tmp,
                                                int(np.trunc((values_to_gen[j, qi_data_idx_to_gen] % 100) / 10) * 10))
                    if lv == 4:
                        if j == 0:
                            tmp = np.nan
                        else:
                            tmp = np.append(tmp, np.nan)

                else:
                    if j == 0:
                        tmp = np.nan
                    else:
                        tmp = np.append(tmp, np.nan)

            return np.c_[values_to_gen[:, 0:qi_data_idx_to_gen], tmp, values_to_gen[:, qi_data_idx_to_gen + 1:]]
    return values_to_gen


def generalize_string(values_to_gen: np.ndarray, qi_string_idx_to_gen: int, level_of_generalization: int):
    """
    Generalizes the string values: 'Job Title' or 'Department' with specified level of generalization
    by appropriate param

    :param values_to_gen: Entire dataset
    :param qi_string_idx_to_gen: 'Job Title' (qi_string_idx_to_gen = 0) or 'Department' (qi_string_idx_to_gen = 1)
    :param level_of_generalization: Specify level of generalization (1 or 2)
    :return: Entire dataset with generalized values
    """
    tmp = []
    if qi_string_idx_to_gen in range(0, 2):
        if level_of_generalization == 1:
            hierarchy = []
            if qi_string_idx_to_gen == 0:
                hierarchy = create_string_generalize_hierarchy(values_to_gen, qi_string_idx_to_gen)
            elif qi_string_idx_to_gen == 1:
                hierarchy = create_string_generalize_hierarchy(values_to_gen, qi_string_idx_to_gen)
                print(values_to_gen[0:10, 1])
            for j in range(values_to_gen.shape[0]):
                for k in range(hierarchy.shape[0]):
                    if values_to_gen[j, qi_string_idx_to_gen] == hierarchy[k, 0]:
                        if j == 0:
                            tmp = hierarchy[k, 1].strip()
                        else:
                            tmp = np.append(tmp, hierarchy[k, 1].strip())
            return np.c_[values_to_gen[:, 0:qi_string_idx_to_gen], tmp, values_to_gen[:, qi_string_idx_to_gen + 1:]]
        elif level_of_generalization == 2:
            for j in range(values_to_gen.shape[0]):
                if j == 0:
                    tmp = '*'
                else:
                    tmp = np.append(tmp, '*')
            return np.c_[values_to_gen[:, 0:qi_string_idx_to_gen], tmp, values_to_gen[:, qi_string_idx_to_gen + 1:]]

    return values_to_gen


def check_strings_occ(data: np.ndarray, idx: int, threshold: int):
    """
    Prints QIs of tuples' that occurs under a certain treshold,
    the number of those tuples,
    the list of QI and the tuples that are over this treshold
    and the list of QI and the tuples that are under this treshold
    :param data: Entire Dataset
    :param idx: The QI passed by the user
    :param threshold: the number of tuples will be highlighted
    """
    [arr, count] = np.unique(data[:, idx], return_counts=True)
    print(list(arr[np.where(count < threshold)]))
    print((arr[np.where(count < threshold)].shape[0]))
    print(list(zip(list(arr[np.where(count > threshold)]), list(count[np.where(count > threshold)]))))
    print(list(zip(list(arr[np.where(count < threshold)]), list(count[np.where(count < threshold)]))))


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
        plt.ylim([0, 100])
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
        plt.ylim([0, 100])
        plt.xlim([1965, 2025])
        plt.grid()
        plt.show()
    elif idx == 5:
        plt.figure()
        plt.hist(data[:, idx], bins=250)
        plt.title(labels[idx] + " distribution")
        plt.ylim([0, 100])
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
    print(occurrences)
    for j in range(len(occurrences)):
        if occurrences[j] < k:
            return False
    return True


def delete_outliers_of_data_before(data: np.ndarray, qi_inspect: int, threshold: int):
    """
    Delete tuples more distant from the average behavior of the Dataset by data
    before applying the generalization
    :param data: Entire Dataset
    :param qi_inspect: The index of the QI
    :param threshold: the number of tuples chosen as bind
    :return: the list of the tuples to be deleted
    """
    idx_to_del = []
    done = False
    for j in range(data.shape[0]):
        if data[j, qi_inspect] < threshold:
            if not done:
                idx_to_del = j
                done = True
            else:
                idx_to_del = np.append(idx_to_del, j)
    return np.delete(data, idx_to_del, axis=0)


def delete_outliers_of_string_before(data: np.ndarray, qi_inspect: int):
    """
    Delete tuples more distant from the average behavior of the Dataset by Job
    before applying the generalization
    :param data: Entire Dataset
    :param qi_inspect: The index of the QI
    :return: the list of the tuples to be deleted
    """
    idx_to_del = []
    done = False
    for j in range(data.shape[0]):
        if data[j, qi_inspect] == 'REAL' or data[j, qi_inspect] == 'STUDENT':
            if not done:
                idx_to_del = j
                done = True
            else:
                idx_to_del = np.append(idx_to_del, j)
    return np.delete(data, idx_to_del, axis=0)


def start_tables_generation(qi_indices=None):
    """
    First step of incognito: creates generalized table for QIs.
    Those are the matrix representation of the nodes composed
    by the generalization levels of the subset of attributes
    and the relative edges that define direct generalization
    hierarchy from less gen. node to the most gen.
    :param qi_indices:
    :return: nodes and edges of the tree in matrix and vector form
    """
    max_level = 0
    bias = [0, 3, 6, 11, 16]
    nodes_table = []
    edges_table = []
    for qi in qi_indices:
        if qi in range(0, 2):
            max_level = 2
        elif qi in range(2, 5):
            max_level = 4

        for level in range(max_level + 1):
            tmp_node = np.append(np.append(level + bias[qi] + 1, qi), level)
            if level == 0 and qi == 0:
                nodes_table = tmp_node
            else:
                nodes_table = np.concatenate((nodes_table, tmp_node))

    nodes_table = np.reshape(nodes_table, (int(nodes_table.shape[0] / 3), 3))
    for j in range(nodes_table.shape[0] - 1):
        if nodes_table[j, 1] == nodes_table[j + 1, 1]:
            tmp_edge = np.append(nodes_table[j, 0], nodes_table[j + 1, 0])
            if j == 0:
                edges_table = tmp_edge
            else:
                edges_table = np.concatenate((edges_table, tmp_edge))

    edges_table = np.reshape(edges_table, (int(edges_table.shape[0] / 2), 2))
    return nodes_table, edges_table


def find_roots(table_of_nodes: np.ndarray, table_of_edges: np.ndarray):
    """
    Finds all the nodes without incoming edges
    :param table_of_nodes: Matrix of generalized nodes
    :param table_of_edges: Matrix of edges related to generalized nodes
    :return: matrix of root nodes
    """
    rts = []
    for j in range(table_of_nodes.shape[0]):
        flag = True
        for k in range(table_of_edges.shape[0]):
            if int(table_of_edges[k, 1]) == int(table_of_nodes[j, 0]):
                flag = False
        if flag:
            if j == 0:
                rts = table_of_nodes[j, :]
            else:
                rts = np.concatenate((rts, table_of_nodes[j, :]))
    rts = np.reshape(rts, (int(rts.shape[0] / table_of_nodes.shape[1]), table_of_nodes.shape[1]))
    return rts


def insert_roots_into_queue(rt: np.ndarray):
    """
    Insert all the nodes that has no direct incoming edges
    into queue
    :param rt: vector of root nodes
    :return: vector queue
    """
    q = []
    for j in range(rt.shape[0]):
        if j == 0:
            q = rt[j, :]
        else:
            q = np.concatenate((q, rt[j, :]))
    q = np.reshape(q, (int(q.shape[0] / rt.shape[1]), rt.shape[1]))
    return q


def generalize_values(data: np.ndarray, qi_indices: np.ndarray, levels: np.ndarray):
    """
    Generalize all values identified by certain specific QIs
    :param data: Entire Dataset
    :param qi_indices: One or a list of QI for generalization operation
    :param levels: Desired level of generalization
    :return: The generalized Dataset
    """
    if np.isscalar(qi_indices):
        if qi_indices in range(0, 2):
            data = generalize_string(data, int(qi_indices), int(levels))
        if qi_indices in range(2, 5):
            data = generalize_data(data, int(qi_indices), int(levels))
    elif qi_indices.shape[0] > 1:
        for j in range(qi_indices.shape[0]):
            if qi_indices[j] in range(0, 2):
                data = generalize_string(data, qi_indices[j], levels[j])
            if qi_indices[j] in range(2, 5):
                data = generalize_data(data, qi_indices[j], int(levels[j]))

    return data


def get_node_indices_and_levels(nd: np.ndarray):
    """
    Retrieve QIs and their generalization level
    :param nd: Matrix of generalized nodes
    :return: vector of QIs and their levels
    """
    indices = []
    lvs = []
    for j in range(1, nd.shape[0]):
        if j == 1:
            indices = nd[j]
            lvs = nd[j + 1]
        elif j % 2 != 0 and j > 1:
            indices = np.append(indices, nd[j])
        elif j % 2 == 0 and j > 2:
            lvs = np.append(lvs, nd[j])
    return indices, lvs


def mark_all_direct_generalizations(id_node: int, marks: np.ndarray, edges: np.ndarray):
    """
    Finds all connected nodes (direct generalizations) by looking at edges matrix
    :param id_node: Identifier of node from wich we have to find direct connected nodes
    :param marks: Vector representing directed generalization nodes
    :param edges: 2D list of connection between nodes
    :return: vector of marked nodes
    """
    for j in range(edges.shape[0]):
        if edges[j, 0] == id_node:
            marks = np.append(marks, edges[j, 1])
    return marks


def insert_direct_generalizations_of_node_into_queue(id_node: int, q: np.ndarray, edges: np.ndarray, nodes: np.ndarray):
    """
    Once found direct generalizations (nodes) of the node given as parameter (ID), then put them in the queue
    :param id_node: Node identifier to start search
    :param q: List of nodes
    :param edges: Connection between nodes
    :param nodes: Matrix of generalized Nodes
    :return:
    """
    for j in range(edges.shape[0]):
        if edges[j, 0] == id_node:
            q = np.concatenate((q, nodes[np.where(edges[j, 1] == nodes[:, 0]), :][0]))
    return q


def graph_generation(nodes: np.ndarray, edges: np.ndarray):
    """
    Refering to Incognito, it is the generation of Candidate Set of Nodes
    where the subset of QI grows till generate Ci with composed of all
    QI chosen for anonimization:
    ex. number of Qi = n
        1st iteration generates generalized nodes composed of 2 QI
        2nd iteration generates generalized nodes composed of 3 QI
        ...
        ith iteration generates generalized nodes composed of n QI

    At each iteration will be generated a set of direct edges
    connecting the new nodes' set

    In the end only nodes satisfing K-anonimity wil be taken and
    as consequence on the edges

    :param nodes: Matrix of generalized nodes
    :param edges: onnection between nodes
    :return: Matrix of nodes satisfing k-an and relative edges
    """
    result_nodes = []
    result_edges = []

    last_index = nodes[-1, 0]

    done = False
    # print(nodes)
    for p in range(nodes.shape[0]):
        for q in range(nodes.shape[0]):
            if list(nodes[p, 1:-2]) == (list(nodes[q, 1:-2])) and nodes[p, nodes.shape[1] - 2] < \
                    nodes[q, nodes.shape[1] - 2]:
                tmp_node = np.append(nodes[p, 1:], np.append(nodes[q, nodes.shape[1] - 2],
                                                             np.append(nodes[q, nodes.shape[1] - 1],
                                                                       np.append(nodes[p, 0],
                                                                                 nodes[q, 0]))))
                if not done:
                    result_nodes = [tmp_node]
                    done = True
                else:
                    result_nodes = np.concatenate((result_nodes, [tmp_node]))

    result_nodes = result_nodes[np.argsort(
        result_nodes[:, [e for e in range(1, result_nodes.shape[1] - 2) if e % 2 != 0]].sum(axis=1)), :]

    result_nodes = np.c_[range(last_index + 1, last_index + 1 + result_nodes.shape[0]), result_nodes]
    # print(result_nodes)

    done = False
    for e in range(edges.shape[0]):
        for f in range(edges.shape[0]):
            for p in range(result_nodes.shape[0]):
                for q in range(result_nodes.shape[0]):
                    if (edges[e, 0] == result_nodes[p, -2] and edges[e, 1] == result_nodes[q, -2] and edges[
                        f, 0] == result_nodes[p, -1] and edges[f, 1] == result_nodes[q, -1]) \
                            or (edges[e, 0] == result_nodes[p, -2] and edges[e, 1] == result_nodes[q, -2] and
                                result_nodes[p, -1] == result_nodes[q, -1]) \
                            or (edges[e, 0] == result_nodes[p, -1] and edges[e, 1] == result_nodes[q, -1] and
                                result_nodes[p, -2] == result_nodes[q, -2]):
                        if not done:
                            result_edges = [[result_nodes[p, 0], result_nodes[q, 0]]]
                            done = True
                        else:
                            result_edges = np.concatenate(
                                (result_edges, [[result_nodes[p, 0], result_nodes[q, 0]]]), axis=0)
    # print(edges)

    # print(result_edges)
    unique_result_edges = list(Counter(str(e) for e in result_edges).keys())
    # print(unique_result_edges)
    final_edges = []
    for k in range(len(unique_result_edges)):
        for j in range(result_edges.shape[0]):
            if str(result_edges[j]) == unique_result_edges[k]:
                if k == 0:
                    final_edges = result_edges[j]
                    break
                else:
                    final_edges = np.concatenate((final_edges, result_edges[j]))
                    break
    final_edges = np.reshape(final_edges, (int(final_edges.shape[0] / 2), 2))
    # print(final_edges.shape[0])
    done = False
    edge_to_remove = []
    for j in range(final_edges.shape[0]):
        for k in range(j + 1, final_edges.shape[0]):
            if final_edges[j, 1] == final_edges[k, 0]:
                if not done:
                    edge_to_remove = [[final_edges[j, 0], final_edges[k, 1]]]
                    done = True
                else:
                    edge_to_remove = np.concatenate((edge_to_remove, [[final_edges[j, 0], final_edges[k, 1]]]))
    # print(edge_to_remove)

    idx_to_remove = []
    done = False
    for j in range(edge_to_remove.shape[0]):
        for k in range(final_edges.shape[0]):
            if list(edge_to_remove[j]) == list(final_edges[k]):
                if not done:
                    idx_to_remove = k
                    done = True
                else:
                    idx_to_remove = np.append(idx_to_remove, k)
    final_edges = np.delete(final_edges, idx_to_remove, axis=0)
    # print(final_edges)
    result_nodes = np.delete(result_nodes, [-1, -2], 1)
    # print(result_nodes)
    return result_nodes, final_edges


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
    #print(fields)
    check_strings_occ(values, 0, 100)

    plot_graphs(generalize_string(values, 0, 1), fields, 0)
    plot_graphs(generalize_string(values, 1, 1), fields, 1)

    plot_graphs(values, fields, 2)
    plot_graphs(values, fields, 3)
    plot_graphs(values, fields, 4)

    values = delete_outliers_of_data_before(values, 2, 19890000)
    values = delete_outliers_of_string_before(generalize_string(values, 0, 1), 0)
    check_strings_occ(generalize_string(values, 0, 1), 0, 100)
    plot_graphs(values, fields, 2)
    plot_graphs(values, fields, 3)

    # Algorithm
    [C, E] = start_tables_generation(qi_idx)
    S = C
    # print(C)

    for i in range(fields.shape[0] - 1):
        S = C
        roots = find_roots(C, E)

        queue = insert_roots_into_queue(roots)
        marked = []

        while queue.shape[0] > 0:
            node = queue[0, :]

            print("-------------------------------------------------------------")
            print("Queue:")
            print(queue)
            print("Extracted node: ", node)

            queue = np.delete(queue, 0, 0)

            if not node[0] in marked:
                idx_node, levels_node = get_node_indices_and_levels(node)
                dataset = generalize_values(values, idx_node, levels_node)
                if check_k_anonymity(dataset, 2, idx_node):
                    print("Respect 2-anonymity")
                    marked = mark_all_direct_generalizations(node[0], np.array(marked, dtype='int'), E)
                else:
                    print("NOT respect 2-anonymity")
                    queue = insert_direct_generalizations_of_node_into_queue(node[0], queue, E, S)
                    S = np.delete(S, np.where(S[:, 0] == node[0]), 0)
                    E = np.delete(E, np.where(E[:, 0] == node[0]), 0)
                    E = np.delete(E, np.where(E[:, 1] == node[0]), 0)
        if i < 4:
            C, E = graph_generation(S, E)
        else:
            print(S)

    for x in range(S.shape[0]):
        idx_node, levels_node = get_node_indices_and_levels(S[x])
        dataset = generalize_values(values, idx_node, levels_node)
        # if not check_k_anonymity(dataset, 4, idx_node):
        #    delete_outliers_after_incognito(dataset, 4, idx_node)

        print(check_k_anonymity(dataset, 2, idx_node))
