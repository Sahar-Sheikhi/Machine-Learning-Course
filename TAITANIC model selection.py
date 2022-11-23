X = train[["Pclass","Sex","Age","Fare","Cabin","Prefix","Q","S","Family"]]
Y = train["Survived"]
X_TEST = test[["Pclass","Sex","Age","Fare","Cabin","Prefix","Q","S","Family"]]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X_TEST =  sc.transform(X_TEST)

# Train-Test-Split
# We perform the train-test-split on the training data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2, random_state=1)
# Making the Models
# 1. K — Nearest Neighbor Algorithm
# The K-Nearest Neighbor algorithm works well for classification if the right k value is chosen. We can select the right k value using a small for-loop that tests the accuracy for each k value between 1 and 20.

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

acc = []

for i in range(1,20):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train,y_train)
    yhat = knn.predict(X_test)
    acc.append(accuracy_score(y_test,yhat))
    print("For k = ",i," : ",accuracy_score(y_test,yhat))

plt.figure(figsize=(8,6))
plt.plot(range(1,20),acc, marker = "o")
plt.xlabel("Value of k")
plt.ylabel("Accuracy Score")
plt.title("Finding the right k")
plt.xticks(range(1,20))
plt.show()


# Our preferred value for k that gives us the highest accuracy is k = 9.
# We can now use this k value for making our model:

KNN = KNeighborsClassifier(n_neighbors = 9)
KNN.fit(X,Y)
y_pred = KNN.predict(X_TEST)
df_KNN = pd.DataFrame()
df_KNN["PassengerId"] = test2["PassengerId"]
df_KNN["Survived"] = y_pred

# 2. Decision Tree Algorithm

from sklearn.tree import DecisionTreeClassifier

depth = [];

for i in range(1,8):
    clf_tree = DecisionTreeClassifier(criterion="entropy", random_state = 100, max_depth = i)
    clf_tree.fit(X_train,y_train)
    yhat = clf_tree.predict(X_test)
    depth.append(accuracy_score(y_test,yhat))
    print("For max depth = ",i, " : ",accuracy_score(y_test,yhat))


plt.figure(figsize=(8,6))
plt.plot(range(1,8),depth,color="red", marker = "o")
plt.xlabel("Depth of Tree")
plt.ylabel("Accuracy Score")
plt.title("Finding the right depth with highest accuracy")
plt.xticks(range(1,8))
plt.show()

clf_tr = DecisionTreeClassifier(criterion="entropy", random_state = 100, max_depth = 3)
clf_tr.fit(X,Y)
pred_tree = clf_tr.predict(X_TEST)
df_TREE = pd.DataFrame()
df_TREE["PassengerId"] = test2["PassengerId"]
df_TREE["Survived"] = pred_tree
df_TREE.head()


# 3. Random Forest Algorithm

from sklearn.ensemble import RandomForestClassifier

clf_forest = RandomForestClassifier(random_state=0)
clf_forest.fit(X_train,y_train)
yhat = clf_forest.predict(X_test)
print("Accuracy for training data : ",accuracy_score(y_test,yhat))
Accuracy for training data :  0.776536312849162
We now save it and submit it to Kaggle

clf_for = RandomForestClassifier(random_state=0)
clf_for.fit(X,Y)
y_forest = clf_for.predict(X_TEST)
df_FOREST = pd.DataFrame()
df_FOREST["PassengerId"] = test2["PassengerId"]
df_FOREST["Survived"] = y_forest
df_FOREST.head()

# Our accuracy is 77.27%

# 4. Support Vector Machine
# We try out the Support Vector Machine Algorithm for this classification problem.

from sklearn.svm import SVC
clf_svm = SVC(gamma='auto')
clf_svm.fit(X_train,y_train)
yhat = clf_svm.predict(X_test)
clf_SVM = SVC(gamma='auto')
clf_SVM.fit(X,Y)
pred_svm = clf_SVM.predict(X_TEST)
df_SVM = pd.DataFrame()
df_SVM["PassengerId"] = test2["PassengerId"]
df_SVM["Survived"] = pred_svm
df_SVM.head()

# Our accuracy is 77.51%

# 5. Naive Bayes Algorithm

from sklearn.naive_bayes import GaussianNB
clf_NB = GaussianNB()
clf_NB.fit(X_train,y_train)
y_hat = clf_NB.predict(X_test)
print("Accuracy for training data : ",accuracy_score(y_test,y_hat))

clf_NB = GaussianNB()
clf_NB.fit(X,Y)
pred_NB = clf_NB.predict(X_TEST)
df_NB = pd.DataFrame()
df_NB["PassengerId"] = test2["PassengerId"]
df_NB["Survived"] = pred_NB
df_NB.head()

# Our accuracy is 72.72%

# 6. Logistic Regression Algorithm

from sklearn.linear_model import LogisticRegression
regr = LogisticRegression(solver='liblinear', random_state=1)
regr.fit(X_train,y_train)
yhat = regr.predict(X_test)
print("Accuracy for training data : ",accuracy_score(y_test,y_hat))

reg = LogisticRegression(solver='liblinear', random_state=1)
reg.fit(X,Y)
y_LR = reg.predict(X_TEST)
df_LR = pd.DataFrame()
df_LR["PassengerId"] = test2["PassengerId"]
df_LR["Survived"] = y_LR
df_LR.head()

# Our accuracy is 76.55%

# 7. Stochastic Gradient Descent Classifier

from sklearn.linear_model import SGDClassifier

clf_SGD = SGDClassifier(loss="squared_loss", penalty="l2", max_iter=4500,tol=-1000, random_state=1)
clf_SGD.fit(X_train,y_train)
yhat = clf_SGD.predict(X_test)
print(accuracy_score(y_test,yhat))

# Training Accuracy
clf_SGD = SGDClassifier(loss="squared_loss", penalty="l2", max_iter=4500, tol=-1000, random_state=1)
clf_SGD.fit(X,Y)
y_SGD = clf_SGD.predict(X_TEST)
df_SGD = pd.DataFrame()
df_SGD["PassengerId"] = test2["PassengerId"]
df_SGD["Survived"] = y_SGD
df_SGD.head()

# Our accuracy is 76.79%

# Final Results
plt.figure(figsize=(8,6))
plt.plot(range(1,8),[KNN_accuracy,TREE_accuracy,FOREST_accuracy,SVM_accuracy,NB_accuracy,LR_accuracy,SGD_accuracy],marker='o')
plt.xticks(range(1,8),['KNN','Decision Tree','Random Forest','SVM','Naive Bayes','Log Regression','SGD'],rotation=25)
plt.title('Accuracy of Various Models')
plt.xlabel('Model Names')
plt.ylabel("Accuracy Score")
plt.show()

