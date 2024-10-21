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
    train_data = load_data("adult/adult_processed.csv") # 30718*14
    test_data = load_data("adult/adult_test_processed.csv") # 15316*14
    x_train = [row[:-1] for row in train_data[1:]] # 30717*13
    y_train = [[row[-1]] for row in train_data[1:]] # 30717*1
    x_test = [row[:-1] for row in test_data[1:]] # 15315*13
    y_test = [[row[-1]] for row in test_data[1:]] # 15315*1
    
    # 2. introduce a model
    dt = DecisionTree(num_bins=8, max_depth=12, min_samples_split=15)
    # 3. train and evaluate
    dt.fit(train_data)
    print("========= Finished the model training =========")
    
    y_pred_train = dt.predict(x_train)
    y_pred_test = dt.predict(x_test)
    
    train_accuracy, train_precision, train_recall, train_f1_score = dt.evaluate(y_train, y_pred_train)
    print("========= Evaluation on training set =========")
    print(f"Accuracy: {train_accuracy}")
    print(f"Precision: {train_precision}")
    print(f"Recall: {train_recall}")
    print(f"F1 Score: {train_f1_score}")
    
    test_accuracy, test_precision, test_recall, test_f1_score = dt.evaluate(y_test, y_pred_test)
    print("========= Evaluation on test set =========")
    print(f"Accuracy: {test_accuracy}")
    print(f"Precision: {test_precision}")
    print(f"Recall: {test_recall}")
    print(f"F1 Score: {test_f1_score}")
    

    # 4. save model
    dt.save_tree("decision_tree.txt")


if __name__ == '__main__':
    main()
    