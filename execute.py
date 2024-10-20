from DecisionTree import DecisionTree
import csv

def load_data(filename):
    """
    load the dataset from a csv file
    """
    with open(filename, 'r') as f:
        data = [] # preprocessed data
        csv_reader = csv.reader(f)
        
        for row in csv_reader:
            data.append(row)
        
    return data

def main():
    # 1. load data
    # Note: the first row of your data must be header
    train_data = load_data("adult/adult_processed.csv")
    test_data = load_data("adult/adult_test_processed.csv")
    # 2. introduce a model
    # you can also define the num_bins using dt = DecisionTree(num_bins=xxx)
    dt = DecisionTree(num_bins=8, max_depth=12, min_samples_split=15)
    # 3. train and evaluate
    dt.fit(train_data)
    
    train = train_data[1:]
    test = test_data[1:]
    train_accuracy = dt.evaluate(train)
    print("========= Evaluation on training set =========")
    print(f"accuracy: {train_accuracy}")
    test_accuracy = dt.evaluate(test)
    print("========= Evaluation on test set =========")
    print(f"accuracy: {test_accuracy}")
    # 4. save model
    dt.save_tree("decision_tree.txt")


if __name__ == '__main__':
    main()
    