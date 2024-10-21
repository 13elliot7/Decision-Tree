import csv

# test file
# input_file = 'D:/work/task/cuhk/cmsc5724/proj/adult/adult.test'
# output_file = 'D:/work/task/cuhk/cmsc5724/proj/adult/adult_test_processed.csv'

#train file
input_file = 'D:/work/task/cuhk/cmsc5724/proj/adult/adult.data'
output_file = 'D:/work/task/cuhk/cmsc5724/proj/adult/adult_processed.csv'

# Define the column headers
headers = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "income"
]

# Read from the .test file and write to a .csv file with necessary adjustments
with open(input_file, 'r') as data_file:
    with open(output_file, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(headers)  # Write the headers

        # Skip the first line which is metadata or a comment
        next(data_file)

        for line in data_file:
            # Strip whitespace and skip empty lines
            line = line.strip()
            if line:
                row = [value.strip() for value in line.split(',')]

                # Check if the number of columns matches our expectations
                if len(row) == 15:
                    # Remove the trailing period from the income field
                    row[-1] = row[-1].rstrip('.')

                    # Remove the "native-country" field (2nd to last)
                    row.pop(-2)

                    # Check if any cell contains '?', if so, skip the line
                    if '?' not in row:
                        csv_writer.writerow(row)

print(f"Conversion complete. CSV file saved at {output_file}")
