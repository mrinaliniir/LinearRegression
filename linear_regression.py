import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt


#df = pd.DataFrame(np.random.randint(0,10,size=(100, 2)), columns=list('Xy')) 
df = pd.read_csv("C:/Users/irmri/OneDrive/Desktop/ex1data1.txt"); # load the data 
df.columns = ['X', 'y'] 										  # assign column names X = features, y = predictor variable

alpha= 0.01 #learning rate
iter = 10000 # iterations
m = len(df)  # total number of examples

x = pd.DataFrame(np.ones(len(df)),columns=["ones"])       # add column of ones 
x['X'] = df['X']
theta = np.array([-1,2], dtype='float')                   # initialize weights
 
 
# Cost Function
def costFunction(x,y,theta):
	predicted = x.dot(theta.T)
	J = np.square(predicted - df.y).sum()/(2*m)
	return J


# Gradient Descent
def gradientDescent(x, y, theta, alpha, iterations):
	for i in range(iterations):
	    h_theta = x.dot(theta.T)
	    for index in range(len(theta)):
	        theta[index] = theta[index] - (alpha/m) * (h_theta-y).dot(x[x.columns[index]])
	return theta
		
# Split the dataset into training and test data
X_train, X_test, y_train, y_test = train_test_split(x, df.y, test_size=0.2)


def trainModel(X_train, y_train):
    print("Training the model...")
    optimized_theta = gradientDescent(X_train, y_train, theta, alpha, iter)  # optimized weights
    return optimized_theta
	
def predictModel(theta, X_train):
	predictions = X_train.dot(theta.T)  # Get predictions
	return predictions
	
# Calculate the mean squared error
def rmse(X_train, y_train, theta):
    predicted_value = predictModel(theta, X_train)
    actual_value = y_train
    mean_error = np.sqrt(np.mean((predicted_value - actual_value)**2))
    return mean_error
	
weights = trainModel(X_train, y_train)
print("********Results for linear regression using current model**********")
RMSE = rmse(X_test, y_test, weights)
print("RMSE :", RMSE) 
print(weights)
print("******************")

model = LinearRegression()
model.fit(X_train, y_train)
print("******** Results for linear regression using scikit learn library **********")
y_pred = model.predict(X_test)
RMSE2 = sqrt(mean_squared_error(y_test, y_pred))
print("RMSE :", RMSE2)
print(model.coef_)
print(model.intercept_)
print("******************")


# Fit the model on train data
plt.scatter(X_train.X, y_train)
plt.plot(X_train.X, model.predict(X_train), color = 'blue')
plt.show()

# Fit the model on test data
plt.scatter(X_test.X, y_test)
plt.plot(X_test.X, model.predict(X_test), color = 'blue')
plt.show()


		




	
	

	





