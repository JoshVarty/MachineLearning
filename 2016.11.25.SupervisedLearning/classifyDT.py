def classify(features_train, labels_train):
    from sklearn.tree import DecisionTreeClassifier
    ### your code goes here--should return a trained decision tree classifer

    clf = DecisionTreeClassifier()
    clf.fit(features_train, labels_train)
    
    return clf