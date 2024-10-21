# Decision Tree Classifier using Hunt's Algorithm
This project implements a decision tree classifier using Hunt's algorithm, optimized with the GINI index for determining splits. The classifier is applied to the **Adult Income Dataset**, with functions to train, predict, and evaluate the model, as well as save the trained dicision tree to a file.

## Features
* Recursive decision tree construction using hunt's algorithm.
* Support for both **categorical** and **numerical** attributes.
* GINI index and GINI gain for finding the best splits.
* Customizable **hyperparameters**:
    * ```max_depth```: Maximum depth of the tree to prevent overfitting.
    * ```min_samples_split```: Minumum number of samples required to perform a split.
    * ```num_bins```: Binning strategy for numeric attributes to reduce the number of potential splits for efficiency.
* Evaluation and prediction on the given dataset.
* Save the trained decision tree in a readable format.
* Encapsulated DecisionTree class.

## Data preprocessing

Read from the adult.data file and write to a .csv file. First, We set the column headers for the CSV file. The headers represent the different attributes in the dataset, such as age, workclass, and others.

```python
# Check if any cell contains '?'
# Remove cells containing '?'
if any('?' in value for value in row):
    continue
```

This checks if any value in the row contains a question mark (`'?'`), which indicates missing or unknown data in the dataset. If any value contains `'?'`, the line is skipped using `continue`. This removes rows with missing data.

```python
# Remove the second to last attribute (native-country)
row.pop(-2)
```

The input dataset contains an attribute called `native-country` which is removed in the CSV output. `row.pop(-2)` removes the second-to-last element from the row. The `native-country` attribute was originally positioned as the second-to-last field in the dataset.

```python
csv_writer.writerow(row)
```

Finally, the processed row is written to the output CSV file.

## Implementation of Hunt's algorithm
Hunt's algorithm builds a decision tree recursively. The main steps are:
### 1. Base Cases:
* If all examples in the current dataset have the **same label**, the algorithm stops, and the label is returned.
* If all examples have the **same attribute** values but different labels, return the majority label.
* If the **maximum depth** is reached or the number of samples is **smaller** than the ```min_samples_split```, return the majority label. (alleviate overfitting)

### 2. Best Split Selection
* For each attribute (both numeric and categorical), the algorithm calculates the GINI GAIN to find the best split.
* For **numerical attributes**, different thresholds (chosen from the sorted attribute values) are tested to split the dataset, and the optimal one is chosen (the smallest one).
* For **categorical attributes**, subsets are created based on the unique attribute values.

### 3. Recursion and Stopping
* The algorithm then recursively applies itself to the subsets created by the best split, building subtrees.
* The process continues until one of the base cases is met.

### 4. Efficiency Optimization Strategy
* ```num_bins``` is introduced to group the numeric values and reduce the number of potential splits, providing efficiency.

## Files
* ```process.py```: Data preprocess, including remove all the records containing missing values and the attribute "native-country".
* ```hunt_algorithm.py```: Core functions to build the decision tree using Hunt's algorithm.
* ```DecisionTree.py```: An encapsulated class of decision tree, including methods for training, predicting, evaluating and outputing the tree to the disk.
* ```execute.py```: A script to train the decision tree on the given Adult dataset, evaluate its performance and save the predictions.
* ```decision_tree.txt```: A file containing the structure of the output decision tree.

## How to Use

## Requirements
* Python 3.x

## Installation
Save the entire project to your disk and navigate to the project directory.

## Dataset
The project uses the **Adult Dataset**. Ensure that the dataset is preprocessed and stored as:

* ```adult/adult_processed.csv```: Training data
* ```adult/adult_test_processed.csv```: Test data

Both files should have a header row.

## Running the project
To train the model, evaluate its performance and get the predictions:

```bash
python execute.py
```

After training, the decision tree will be saved to ```decision_tree.txt```

## Customization
You can modify the parameters of the decision tree (e.g., max depth, minumum samples for spliting, bining strategy for numeric attributes) in ```execute.py``` by changing the following line:

```python
dt = DecisionTree(num_bins=8, max_depth=12, min_samples_split=15)
```

## Output
* 





