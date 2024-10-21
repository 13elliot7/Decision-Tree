from DecisionTree import DecisionTree
import csv
import sys

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


def save_data(filename, data, headers):
    """
    save the data to a csv file
    """
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for row in data:
            writer.writerow(row)
    print(f"Data has been saved.")


def main(num_bins=100, max_depth=10, min_samples_split=25):
    # 1. load data
    # Note: the first row of your data must be header
    train_data = load_data("adult/adult_processed.csv") # 30718*14
    test_data = load_data("adult/adult_test_processed.csv") # 15316*14
    x_train = [row[:-1] for row in train_data[1:]] # 30717*13
    y_train = [[row[-1]] for row in train_data[1:]] # 30717*1
    x_test = [row[:-1] for row in test_data[1:]] # 15315*13
    y_test = [[row[-1]] for row in test_data[1:]] # 15315*1
    
    # 2. Initialize a decision tree model with some hyperparameters
    # headers = ['num_bins', 'max_depth', 'min_samples_split', 'train_accuracy', 'test_accuracy']
    # res = []
    # for num_bins in range(10, 20, 5):
    #     for max_depth in range(4, 22, 2):
    #         for min_samples_split in range(20, 55, 5):
    #             dt = DecisionTree(num_bins=num_bins, max_depth=max_depth, min_samples_split=min_samples_split)
    #             dt.fit(train_data)
    #             print("========= Finished the model training =========")
    
    #             y_pred_train = dt.predict(x_train)
    #             y_pred_test = dt.predict(x_test)
    
    #             train_accuracy, train_precision, train_recall, train_f1_score = dt.evaluate(y_train, y_pred_train)
    #             print(f"========= num_bins={num_bins}, max_depth={max_depth}, min_samples_split={min_samples_split} =========")
    #             print("========= Evaluation on Training Set =========")
    #             print(f"Accuracy: {train_accuracy}")
    #             print(f"Precision: {train_precision}")
    #             print(f"Recall: {train_recall}")
    #             print(f"F1 Score: {train_f1_score}")
    
    #             test_accuracy, test_precision, test_recall, test_f1_score = dt.evaluate(y_test, y_pred_test)
    #             print("========= Evaluation on Evaluation Set =========")
    #             print(f"Accuracy: {test_accuracy}")
    #             print(f"Precision: {test_precision}")
    #             print(f"Recall: {test_recall}")
    #             print(f"F1 Score: {test_f1_score}")
    #             res.append([num_bins, max_depth, min_samples_split, train_accuracy, test_accuracy])
    # save_data("output/combination.csv", res, headers)        
                
    dt = DecisionTree(num_bins=num_bins, max_depth=max_depth, min_samples_split=min_samples_split) # 10 10 25
    
    # 3. train and evaluate
    dt.fit(train_data)
    print("========= Finished the model training =========")
    
    y_pred_train = dt.predict(x_train)
    y_pred_test = dt.predict(x_test)
    
    train_accuracy, train_precision, train_recall, train_f1_score = dt.evaluate(y_train, y_pred_train)
    print("========= Evaluation on Training Set =========")
    print(f"Accuracy: {train_accuracy}")
    print(f"Precision: {train_precision}")
    print(f"Recall: {train_recall}")
    print(f"F1 Score: {train_f1_score}")
    
    test_accuracy, test_precision, test_recall, test_f1_score = dt.evaluate(y_test, y_pred_test)
    print("========= Evaluation on Evaluation Set =========")
    print(f"Accuracy: {test_accuracy}")
    print(f"Precision: {test_precision}")
    print(f"Recall: {test_recall}")
    print(f"F1 Score: {test_f1_score}")
    

    # 4. save model and predictions
    dt.save_tree("output/decision_tree.txt")
    data = test_data[1:]
    for i in range(len(data)):
        data[i].append(y_pred_test[i][0])
    headers = [
        "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
        "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
        "hours_per_week", "income", "predict"
    ]
    save_data("output/test_pred.csv", data, headers)


if __name__ == '__main__':
    argv = sys.argv
    if len(argv) > 1:
        num_bins = argv[1]
        max_depth = argv[2]
        min_samples_split = argv[3]
        main(num_bins=num_bins, max_depth=max_depth, min_samples_split=min_samples_split)
    else:
        main()
    