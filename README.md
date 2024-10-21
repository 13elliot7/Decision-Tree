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

In our project, we need to clean the dataset, which consists of 13 attributes plus a label (the 14th index). To accomplish this, we utilize a data cleaning process to read the dataset line by line and eliminate unnecessary data.

```python
# Define the column headers
headers = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "income"
]
```

The process takes the training set or the test set as input, and uses `line.strip()` to remove any leading or trailing whitespace from each line. After cleaning the whitespace, we split each line into individual attributes using a comma delimiter (`', '`). By iterating through each split using a `for` loop, we effectively process each value in the line.

```python
for line in data_file:
            # Strip whitespace and skip empty lines
            line = line.strip()
            if line:
                row = [value.strip() for value in line.split(',')]
```

Since we need to remove the attribute "native-country," which is located at the second-to-last index of each line, we use `row.pop(-2)` to remove it effectively. This approach successfully removes the "native-country" attribute.

&#x20;we remove all records containing `?`, which indicate missing values. To achieve this, we use an `if` conditional statement to check for `?` in the values. If any value contains `?`, we skip that line and only keep the valid data.

```python
# Check if any cell contains '?' skip
if any('?' in value for value in row):
    continue
```

After iterating through all lines, we obtain all the valid, cleaned records. The cleaning process then returns this cleaned dataset. In summary, we have successfully completed the data cleaning process and obtained the required valid data.

```python
csv_writer.writerow(row)
```



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

After training, the decision tree will be saved to ```output/decision_tree.txt```

## Customization
You can modify the path of dataset and parameters of the decision tree (e.g., bining strategy for numeric attributes, max depth, minumum samples for spliting) following the example:

```python
python execute.py 10 20 30
```

## Output
* ```output/decision_tree.txt```: A file containing the structure of the output decision tree.
* ```output/test_pred.csv```: A file containing the predictions on evaluation set using trained decision tree.

The output decision tree will look like this:
```
|--Attribute Education
	|--Bachelors
		|--Attribute occupation
			|--Exec-managerial
				|--Attribute marital_status
					|--Married-civ-spouse
						|--Attribute workclass
							|--Self-emp-not-inc
								|--Attribute race
									|--White
										|--Attribute relationship
											|--Husband
												|--Attribute age <= 26.0
													|--True
														|--Label <=50K
```

## Evaluation

This evaluation process calculates evaluation metrics including accuracy, precision, recall, and F1 score to quantify the model's predictive capabilities.

The evaluation function takes two sets of input: the actual labels (`y_actual`) and the predicted labels (`y_pred`). It first extracts the relevant label values to facilitate metric calculations. The function begins by computing accuracy, which is the proportion of correct predictions among the total predictions. To do this, it compares each actual label with its corresponding predicted label and counts the number of matches.

```python
			y_actual = [label[0] for label in y_actual]
            y_pred = [label[0] for label in y_pred]
            
            # Calculate accuracy
            correct_num = sum(1 for a, p in zip(y_actual, y_pred) if a == p)
            accuracy = correct_num / len(y_actual)
```

In our implementation, we define two classes for the income labels: `>50K` (positive class) and `<=50K` (negative class). We then initialize variables to track true positives (TP), false positives (FP), and false negatives (FN). These counts are essential for calculating precision, recall, and the F1 score.

```python
            # Define the positive and negative classes
            POSITIVE = '>50K'
            NEGATIVE = '<=50K'

            # Initialize counts
            TP = 0  # True Positives
            FP = 0  # False Positives
            FN = 0  # False Negatives
```

The function iterates through the actual and predicted labels, incrementing TP when both actual and predicted labels indicate `>50K`, FP when an actual `<=50K` is predicted as `>50K`, and FN when an actual `>50K` is predicted as `<=50K`.

```python
            # Calculate TP, FP, and FN
            for actual, pred in zip(y_actual, y_pred):
                    if actual == POSITIVE and pred == POSITIVE:
                        TP += 1  # True Positive
                    elif actual == NEGATIVE and pred == POSITIVE:
                        FP += 1  # False Positive
                    elif actual == POSITIVE and pred == NEGATIVE:
                        FN += 1  # False Negative
```

After calculating TP, FP, and FN, the function determines precision (the ratio of true positives to the sum of true and false positives), recall (the ratio of true positives to the sum of true positives and false negatives), and the F1 score (a harmonic mean of precision and recall). 

```python
            # Calculate Precision, Recall, and F1 Score
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
```

In summary, this evaluation method provides a comprehensive assessment of the model's effectiveness by using standard classification metrics, enabling us to understand its strengths and weaknesses in predicting income categories.

The best model performance is shown below:

```
========= Finished the model training =========
========= Evaluation on Training Set =========
Accuracy: 0.8462414949376567
Precision: 0.7345728482128546
Recall: 0.5990849673202614
F1 Score: 0.6599467204262366
========= Evaluation on Evaluation Set =========
Accuracy: 0.8180215475024486
Precision: 0.6738779463997416
Recall: 0.5574252136752137
F1 Score: 0.6101447156848414
```





