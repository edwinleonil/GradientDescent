# Gradient decent implementation

import csv
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  

data = np.genfromtxt("student_scores.csv", delimiter=",", skip_header=1)	
X = data[:,0]
Y = data[:,1]
length = data.shape[0]

m = 0 # initialising the slope
c = 20 # initialising the intercept
b = c # to set intercept axis
learning_rate = 0.005 # try 0.001 to 0.04, 0.05 , 0.06, 0.07, 0.08 
# (0.07 produces oscilation)
slope = []
slope_change=[]
intercept = []
cost_f = []
iteration = []
fig, axs = plt.subplots(2,2,figsize=(10,8))
N = 150

delay = 1
for i in range(0,N):
	iteration.append(i)
	print(i)
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
	# cost_f = (sum_error/length)
	cost_f.append(sum_error/length) # storing result of cost function per iteration
	# check for a change on the slope and intercept change with the number of iterations
	# to see if it converges to a value
	slope.append(m)
	slope_change.append(delta_m)
	intercept.append(c)
	# Prediction for each iteration
	regression_y = []
	for x in X:
		y = m*x + c
		regression_y.append(y)

	axs[0,0].cla()
	axs[0,0].plot(X,regression_y)
	axs[0,0].plot(X,Y,'o')
	axs[0,0].set_xlabel("Hours Studied")
	axs[0,0].set_ylabel("Student Score & y=m*x+c")
	axs[0,0].set_ylim((0,100))
	axs[0,0].set_xlim((0,11))

	axs[0,1].cla()
	axs[0,1].plot(cost_f)
	axs[0,1].set_ylabel("Error value")
	axs[0,1].set_xlabel("Iteration")
	axs[0,1].set_ylim((0,1000))
	axs[0,1].set_xlim((0,N))

	axs[1,0].cla()
	axs[1,0].plot(slope,'tab:orange')
	axs[1,0].set_xlabel("Iteration")
	axs[1,0].set_ylabel("Slope")
	axs[1,0].set_xlim((-5,N))

	axs[1,1].cla()
	axs[1,1].plot(intercept,'tab:green')
	axs[1,1].set_xlabel("Iteration")
	axs[1,1].set_ylabel("Intercept")
	axs[1,1].set_ylim((-b,b))
	axs[1,1].set_xlim((0,N))
	delay = delay*0.5
	plt.pause(delay)
print('error:', sum_error/length)
print('Slope change:', delta_m)
print('Slope:', m)
print('intercept:',c)
plt.show()