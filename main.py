import numpy as np
from sklearn import datasets, linear_model,cross_validation
from sklearn.svm import SVR
from numpy import genfromtxt,dot,mean
from sklearn.ensemble import GradientBoostingRegressor


training_data = genfromtxt('mnist_train.csv', delimiter=',')
training_dataX=training_data[1:1000,1:28*28]
training_dataY=training_data[1:1000,[0]].ravel()

test_data = genfromtxt('mnist_test.csv', delimiter=',')
test_dataX=training_data[1:1000,1:28*28]
test_dataY=training_data[1:1000,[0]].ravel()


clf = linear_model.SGDClassifier()
clf.fit(training_dataX, training_dataY)
print clf.score(test_dataX,test_dataY)

# training_dataX, test_dataX, training_dataY, test_dataY = cross_validation.train_test_split(training_dataX,
#  training_dataY, test_size=0.2, random_state=1)
# gb_regr=GradientBoostingRegressor(n_estimators=750, learning_rate=0.03,
# max_depth=2, random_state=0, loss='ls').fit(training_dataX, training_dataY)
# bg_regr=DecisionTreeRegressor(random_state=0).fit(training_dataX, training_dataY)
# print gb_regr.score(test_dataX,test_dataY)




# test_data = genfromtxt('TestX.csv', delimiter=',',skip_header=1)
# test_data=oneHot(test_data)
# # test_data=enc.fit_transform(test_data).toarray()
# test_dataY=gb_regr.predict(test_data)
# avg=mean(training_dataY)
# test_dataY=np.where(test_dataY >= 0, test_dataY, avg)
# f1=open('output.csv', 'w+')
# print>>f1,"id,cnt"
# for i in range(len(test_data)):
# 	print >>f1, "%d,%d"%(i,test_dataY[i])

