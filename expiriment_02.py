# Decision Tree Classifier
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
import pandas as pd
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test_v2.csv')


columns = ["cost"]

labels = train["cost"].values
features = train[list(columns)].values

dtree = DecisionTreeClassifier(criterion = "entropy", min_samples_leaf = 500)

score = cross_val_score(dtree, features, labels, n_jobs=-1).mean()



dtree.fit(train,test)


model = DecisionTreeClassifier()
model.fit(train.car_value,train.cost)
print(model)

# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)


# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))