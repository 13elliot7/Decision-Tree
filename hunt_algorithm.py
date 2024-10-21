from collections import Counter
import csv
import sys

sys.setrecursionlimit(2000)

def load_data(filename):
    """
    load the dataset from a csv file
    """
    with open(filename, 'r') as f:
        data = [] # preprocessed data
        csv_reader = csv.reader(f)
        next(csv_reader) # skip the first row
        
        for row in csv_reader:
            data.append(row)
        
    return data


def all_attributes_same(data):
    """
    check if all the attributes in the dataset are identical
    """
    for row in data:
        if row[:-1] != data[0][:-1]:
            return False
    return True


def max_majority_label(labels):
    """
    return the max majority value of labels
    """
    return max(set(labels), key=labels.count)


def split_data(data, attr_index):
    """
    split data based on the given categorical attribute

    Args:
        data: the dataset needed to be splited
        attr_index (int): the index of the attribute to be splited

    Returns:
        object: splited subsets
    """
    subsets = {}
    
    for row in data:
        attr_value = row[attr_index]
        if attr_value not in subsets:
            subsets[attr_value] = []
        subsets[attr_value].append(row)
    
    return subsets


def split_numeric_data(data, attr_index, threshold):
    """
    split data into two subsets based on the given numeric attribute threshold

    Args:
        data: the dataset needed to be splited
        attr_index: target attribute index for split
        threshold: the split threshold for numeric attribute

    Returns:
        the splited subsets
    """
    subset_left = [row for row in data if float(row[attr_index]) <= threshold]
    subset_right = [row for row in data if float(row[attr_index]) > threshold]
    return subset_left, subset_right


def gini_gain(data, subsets, target_index):
    """
    calculate the gini gain of the split

    Args:
        data: the dataset needed to be splited
        subsets: the splited dataset
        target_index (int): the index of the label column

    Returns:
        the gini gain of a split
    """
    total = len(data)
    gini_split = 0
    for subset in subsets:
        if len(subset) == 0:
            continue
        labels = [row[target_index] for row in subset]
        size = len(subset)
        gini = sum([labels.count(label) / size ** 2 for label in set(labels)])
        weight = size / total
        gini_split += (1 - gini) * weight
        
    return gini_split

def best_split(data, target_index, num_bins=200):
    """
    find the best split using GINI INDEX(for numeric attribute, also need to find a best split threshold)

    Args:
        data: dataset needed to be splited
        target_index (int): index of label column
        num_bins (int): number of bins to use for numeric attributes, for dealing with the situation that there is too many potential splits

    Returns:
        int: the index of the best attribute to split
    """
    optimal_gini = float('inf')
    res = None # store the best attribute and its threshold for numeric attribute
    # best_attr = None
    n_attr = len(data[0]) - 1
    
    for attr_index in range(n_attr):
        try:
            float(data[0][attr_index])
            # numeric attribute
            sorted_values = sorted(set(float(row[attr_index]) for row in data))
            # Bin the values to reduce the number of potential splits
            if len(sorted_values) > num_bins:
                step_size = len(sorted_values) // num_bins
                sorted_values = sorted_values[::step_size]
            # consider all the value as a potential threshold except the last sorted_value
            for i in range(len(sorted_values) - 1):
                potential_threshold = sorted_values[i]
                subset_left, subset_right = split_numeric_data(data, attr_index, potential_threshold)
                gini_split = gini_gain(data, [subset_left, subset_right], target_index)
                if gini_split < optimal_gini:
                    optimal_gini = gini_split
                    res = (attr_index, potential_threshold)
        except ValueError:
            # categorical attribute
            subsets = split_data(data, attr_index)
            gini_split = gini_gain(data, subsets.values(), target_index)
            if gini_split < optimal_gini:
                optimal_gini = gini_split
                res = (attr_index, None)
    
    return res
    
    
def build_tree(data, target_index, attribute_names, depth=0, max_depth=None, min_samples_split=2, num_bins=200):
    """
    Recursively build the Hunt's decision tree

    Args:
        data
        target_index (int): the index of the label column
        attribute_names
        depth(int): current depth of the tree
        max_depth(int): maximum depth of the tree
        min_samples_split(int): minimum number of samples required to do the split        

    Returns:
        the decision tree
    """
    labels = [row[target_index] for row in data]
    
    # stop condition 1: if all the objects in data share the same label, return the label
    if len(Counter(labels)) == 1:
        return labels[0]
    
    # TODO: stop condition 2: if all the objects in data have the same attribute values, return the majority label
    if(all_attributes_same(data)):
        return max_majority_label(labels)
    
    # TODO: delete stop condition 3: if no more attributes to split, return the majority label
    # if len(data[0]) == 1:
    #     return max_majority_label(labels)
    
    #stop condition 3: if depth is reached or not enough samples, return majority label(alleviate the overfitting)
    if (max_depth is not None and depth >= max_depth) or len(data) < min_samples_split:
        return max_majority_label(labels)
    
    # find the best attribute to split using GINI INDEX
    best_attr, threshold = best_split(data, target_index, num_bins)
    
    if best_attr is None:
        # if no more attributes to split, return the majority label
        return max_majority_label(labels)
    
    if threshold is None:
        # categorical attribute
        term = attribute_names[best_attr]
        tree = {f"Attribute {term}": {}}
        subsets = split_data(data, best_attr)
        for attr_value, subset in subsets.items():
            if subset:
                tree[f"Attribute {term}"][attr_value] = build_tree(subset, target_index, attribute_names, depth + 1, max_depth, min_samples_split, num_bins)
            else:
                tree[f"Attribute {term}"][attr_value] = max_majority_label(labels)
    else:
        # numeric attribute
        term = attribute_names[best_attr]
        tree = {f"Attribute {term} <= {threshold}": {}}
        subset_left, subset_right = split_numeric_data(data, best_attr, threshold)
        tree[f"Attribute {term} <= {threshold}"]["True"] = build_tree(subset_left, target_index, attribute_names, depth + 1, max_depth, min_samples_split, num_bins)
        tree[f"Attribute {term} <= {threshold}"]["False"] = build_tree(subset_right, target_index, attribute_names, depth + 1, max_depth, min_samples_split, num_bins)
    
    return tree


def write_tree(tree, depth, file=None):
    """
    write the decision tree into file and output it

    Args:
        tree (dict): the final decision tree generated with Hunt's algorithm
        depth (int): the depth of the tree
        file (string, optional): file path for saving the decision tree. Defaults to None.
    """
    if file is None:
        return 
    indent = "\t" * depth
    # condition 1: the tree is a label
    if not isinstance(tree, dict):
        line = f"{indent}|--Label {tree}\n"
        # print(line, end="")
        file.write(line)
    
    # traverse the tree
    for key, subtree in tree.items():
        line = f"{indent}|--{key}\n"
        # print(line, end="")
        file.write(line)
        if isinstance(subtree, dict):
            write_tree(subtree, depth + 1, file)  # if it's a subtree, recurse
        else:
            # if it's a label, write it
            line = f"{indent}\t|--Label {subtree}\n"
            # print(line, end="")
            file.write(line)  