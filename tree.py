import pandas as pd
import numpy as np

class Node():
    """
    node in a decision tree.
    """

    def __init__(self, feature=None, category=None, left=None, right=None, gain=None, depth=None, value=None):
        """
        Initializes a new instance of the Node class
        """
        self.feature = feature
        self.category = category
        self.left = left
        self.right = right
        self.gain = gain
        self.depth = depth
        self.value = value

class DecisionTreeCategory:

    def __init__(self, min_samples=2, max_depth=2):
        """
        Constructor for DecisionTree class
        """
        self.min_samples=2
        self.max_depth = max_depth

    def fit(self, X, y):
        dataset = pd.concat([X, y], axis=1)
        self.root = self.build_tree(dataset)
    
    def split_data(self, dataset, feature, category):
        """
        Splits the given dataset into two datasets based on the given feature and category value of this feature.
        """
        left_dataset = dataset[dataset[feature] == category]
        right_dataset = dataset[dataset[feature] != category]

        if left_dataset.empty:
            left_dataset = pd.DataFrame([0])
        if right_dataset.empty:
            right_dataset = pd.DataFrame([0])
        
        return left_dataset, right_dataset
    
    def entropy(self, y):
        """
        Computes the entropy of label values
        """
        entropy = 0
        labels = y.unique()
        for label in labels:
            label_examples = y[y == label]
            pl = len(label_examples) / len(y)
            entropy += -pl * np.log2(pl)

        return entropy

    def gini(self, y):
        pass
           
    def information_gain(self, parent, left, right):
        """
        Computes the information gain from splitting the parent dataset into two datasets
        """
        information_gain = 0
        
        parent_entropy = self.entropy(parent)
        
        weight_left = len(left) / len(parent)
        weight_right= len(right) / len(parent)
        
        entropy_left, entropy_right = self.entropy(left), self.entropy(right)
        
        weighted_entropy = weight_left * entropy_left + weight_right * entropy_right
        
        information_gain = parent_entropy - weighted_entropy
        
        return information_gain


    def best_split(self, dataset):
        """
        Finds the best split for the given dataset
        """
        best_split = {'gain': -1, 'feature': None, 'category': None, "left_dataset": None, "right_dataset": None}

        for col in dataset.columns[:-1]:
            for category in dataset[col].unique():
                left_dataset, right_dataset = self.split_data(dataset, col, category)
                if len(left_dataset) and len(right_dataset):
                    y, left_y, right_y = dataset.iloc[:, -1], left_dataset.iloc[:, -1], right_dataset.iloc[:, -1]
                    information_gain = self.information_gain(y, left_y, right_y)
                    if information_gain > best_split["gain"]:
                        best_split["feature"] = col
                        best_split["category"] = category
                        best_split["left_dataset"] = left_dataset
                        best_split["right_dataset"] = right_dataset
                        best_split["gain"] = information_gain
    
        return best_split

    
    def build_tree(self, dataset, current_depth=0):
        """
        Builds a decision tree from the given dataset
        """
        if current_depth < self.max_depth and len(dataset) > self.min_samples:
            best_split = self.best_split(dataset)
            if best_split["gain"]:
                left_node = self.build_tree(best_split["left_dataset"], current_depth + 1)
                right_node = self.build_tree(best_split["right_dataset"], current_depth + 1)
                
                return Node(best_split["feature"], best_split["category"],
                            left_node, right_node, best_split["gain"], current_depth)

        leaf_value = dataset.iloc[:,-1].mode()[0]

        return Node(value=leaf_value, depth=current_depth)


    def predict(self, X):
        """
        Predicts the class labels for each instance in the feature matrix X
        """
        predictions = []
       
        for i in range(len(X)):
            prediction = self.make_prediction(X.iloc[i], self.root)
            predictions.append(prediction)
        
        np.array(predictions)
        return predictions
    
    def make_prediction(self, x, node):
        """
        Traverses the decision tree to predict the target value for the given feature vector
        """
        if node.value != None: 
            return node.value
        else:
            feature = x[node.feature]
            if feature == node.category:
                return self.make_prediction(x, node.left)
            else:
                return self.make_prediction(x, node.right)

    def print_tree(self, node=None, direction=''):
        """
        Prints decision tree
        """
        if node is None:
            node = self.root
        indent = '-----' * node.depth
        if node.value is not None:
            print(f"{indent}{direction}{node.value}")
        else:
            print(f"{indent}{direction}{node.feature}={node.category}?")
            self.print_tree(node.left, '(T) ')
            self.print_tree(node.right,'(F) ')



            
        






    





        

