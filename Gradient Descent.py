# Gradient decent implementation
#%%
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
data = np.genfromtxt("student_scores.csv", delimiter=",", skip_header=1)	
X = data[:,0]
Y = data[:,1]
length = data.shape[0]
#%%
m = 5 # initialising the slope
c = 10 # initialising the intercept
learning_rate = 0.004 # try 0.001 to 0.04, 0.05 , 0.06, 0.07, 0.08 
# (0.07 produces oscilation)
delta_slope = []
delta_intercept = []
cost_f = []

plt.figure(1)
plt.plot(X,Y,'o')
plt.title("Implementing Gradient Descent")
plt.xlabel("Hours Studied")
plt.ylabel("Student Score")

for i in range(0,100):
    # gradient descent
	sum_error = 0
	for point in data:
		x = point[0]
		y_actual = point[1] 
		y_prediction = m*x + c  
		# Part of cost function or loss function
		error = y_prediction - y_actual 
		# the derivative including the learning rate (step zise)
		delta_m = error * x * learning_rate
		delta_b = error * learning_rate
		# updating the slope and the intercept
		m = m - delta_m
		c = c - delta_b
		sum_error += error*error # total squared error per iteration
	# The goal of any Machine Learning Algorithm 
	# is to minimize the cost Function
	cost_f.append(sum_error/length) # storing result of cost function per iteration
	# check for a change on the slope and intercept change with the number of iterations
	# to see if it converges to a value
	delta_slope.append(delta_m)
	delta_intercept.append(delta_b)
	# Prediction for each iteration
	regression_y = []
	for x in X:
		y = m*x + c
		regression_y.append(y)

plt.figure(1)
plt.plot(X,Y,'o')
plt.plot(X,regression_y)
# Plotting the cost function
plt.figure(2)
plt.plot(cost_f)
plt.title("Error")
plt.xlabel("Iteration")
plt.ylabel("Error value")	
# plotting the slope change
plt.figure(3)
plt.plot(delta_slope)
plt.title("Slope")
plt.xlabel("Iteration")
plt.ylabel("Slope change value")	
# plotting the intercept change
plt.figure(4)
plt.plot(delta_intercept)
plt.title("Intercept")
plt.xlabel("Iteration")
plt.ylabel("Intercept change value")	
plt.show	

# %%
