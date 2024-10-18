import csv

# Specify file paths
# Input and output file paths
input_file = 'D:/work/task/cuhk/cmsc5724/proj/adult/adult.data'
output_file = 'D:/work/task/cuhk/cmsc5724/proj/adult/adult_processed.csv'

# Define the column headers according to the dataset
headers = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "income"
]

# Read from the .data file and write to a .csv file
with open(input_file, 'r') as data_file:
    with open(output_file, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(headers)  # Write the headers

        for line in data_file:
            # Strip whitespace and skip empty lines
            line = line.strip()
            if line:
                row = [value.strip() for value in line.split(',')]
                # Check if any cell contains '?', if so, skip the line
                # Remove cells containing '?'
                if any('?' in value for value in row):
                    continue
                # Remove the second to last attribute (native-country)
                row.pop(-2)
                csv_writer.writerow(row)

print(f"Conversion complete. CSV file saved at {output_file}")

