# Decision-Tree

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
>>>>>>> 57c23bd23506058464e1e7e89ed12bde84fb1f28









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







