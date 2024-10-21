<<<<<<< HEAD
# Decision-Tree

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
=======
# Decision-Tree

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
>>>>>>> 57c23bd23506058464e1e7e89ed12bde84fb1f28







## Evaluation

