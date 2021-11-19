import pandas as pd

col_names=['passengerId', 'pClass', 'sex', 'age', 'sibSp', 'parch','survived']
df=pd.read_csv("titanic.csv", names=col_names).iloc[1:]
print(df.head())

features = ['passengerId', 'pClass', 'sex', 'age', 'sibSp', 'parch','survived']
X=df[features]
y=df.label


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

#splitting data in training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

#Initialising the Decision Tree Model
clf = DecisionTreeClassifier()

#Fitting the data into the model
clf = clf.fit(X_train,y_train)

#Calculating the accuracy of the model

y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

dot_data=StringIO()
export_graphviz(clf, out_file=dot_data, filled=True,rounded=True,special_characters=True, feature_names=features, class_names=['0','1'])
print(dot_data.getvalue())

graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('titanic.png')
Image(graph.create_png())