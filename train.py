import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression
from LogisticRegression import LogisticRegression

# X, y = datasets.make_regression(n_samples=100,n_features= 1,noise=20 , random_state=4)
# X_train , X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1234)


# reg = LinearRegression(alpha=0.01 , n_iters=500)
# reg.fit(X_train,y_train)
# pred = reg.predict(X_test)
# mse = reg.mse(y_test,pred)

# print(f"mse = {mse}")

# pred_line = reg.predict(X)
# fig = plt.figure(figsize=(6,3))
# plt.scatter(X[:,0], y ,color="r",marker="o",s= 30)
# plt.plot(X,pred_line, color="b" , linewidth = 2,label= 'predictions')
# plt.show()



bc = datasets.load_breast_cancer()
X, y = bc.data ,bc.target
X_train , X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1234)


clf = LogisticRegression(alpha=0.01)
clf.fit(X_train,y_train)
pred = clf.predict(X_test)
acc = clf.accuracy(y_test,pred)

print(f"accuracy = {acc}")