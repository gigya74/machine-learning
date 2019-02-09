import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

#constants
alpha = 0.01
Lambda = 1
epslon = 0.000001
filename = 'C:\ml_assignment\data.txt'





colnames = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16']
data = pd.read_csv(filename, delimiter='\s+', names=colnames, skiprows=[0])

print(data.shape)
data.head()
# -------------------------------------------------------------
# feature normalization
for column in data:
    min = data[column].min()
    max = data[column].max()
    data[column] = data[column].apply(lambda x: (x - min) / (max - min))
A1 = data['A1'].values
A2 = data['A2'].values
A3 = data['A3'].values
A4 = data['A4'].values
A5 = data['A5'].values
A6 = data['A6'].values
A7 = data['A7'].values
A8 = data['A8'].values
A9 = data['A9'].values
A10 = data['A10'].values
A11 = data['A11'].values
A12 = data['A12'].values
A13 = data['A13'].values
A14 = data['A14'].values
A15 = data['A15'].values

m = len(data['A1'].values)
x0 = np.ones(m)
X = np.array([x0, A1, A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15])
# Initial Coefficients
theta = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
y = np.array(A15)

#-------------------------------------------------implementation for lasso ---------------------------------------------------------
def regCost_function_lasso(X, y, theta, dt):
    global Lambda

    sq_diff = np.sum((y - X.T.dot(theta)) ** 2)

    term2 = (Lambda * np.sum(abs(theta[1:]))) / (2 * (len(dt)))
    term1 = (1 * sq_diff) / (2 * len(dt))


    return (term1 + term2)


def gradientDesc_lasso(X, y, dt, alpha, theta):

    iters = list()

    num_iter = 0
    m = len(y)
    cost_history = list()
    for i in range(0, 300000):
        num_iter += 1

        y_hat = np.dot(X.T,theta)

        term2 = Lambda * theta/m*abs(theta)

        #theta = theta - alpha * (term1 + term2)
        theta = theta - alpha * (1.0/m) *(np.dot(X,y_hat-y)+term2)
        curr_cost = regCost_function(X, y, theta, dt)
        #print("curr_cost ",curr_cost)
        cost_history.append(curr_cost)
        iters.append(num_iter)
        if (num_iter > 1):
            prev_cost = cost_history[i-1]

            delta_cost = (abs((prev_cost - curr_cost)) * 100 / prev_cost)
            #print(delta_cost, " ", epslon)
            if (delta_cost < epslon):
                print("converged!!! ", num_iter)
                iter = False
                break;

    return cost_history,iters,theta

#-------------------------------------------------implementation for quadratic ---------------------------------------------------------

def regCost_function(X, y, theta, dt):
    global Lambda

    sq_diff = np.sum((y - X.T.dot(theta)) ** 2)

    term2 = (Lambda * np.sum(theta[1:] ** 2)) / (2 * (len(dt)))
    term1 = (1 * sq_diff) / (2 * len(dt))


    return (term1 + term2)

def gradientDesc(X, y, dt, alpha, theta):

    iters = list()
    #print(Lambda," ",epslon)
    num_iter = 0
    m = len(y)
    cost_history = list()
    for i in range(0, 300000):
        num_iter += 1
        #loss = X.T.dot(theta) - y
        y_hat = np.dot(X.T,theta)
        #temp = X.dot(loss)

        #term1 = (1/m) * np.sum(temp)
        #print(theta)
        term2 = (Lambda * theta) / m
        #theta = theta - alpha * (term1 + term2)
        theta = theta - alpha * (1.0/m) *(np.dot(X,y_hat-y)+term2)
        curr_cost = regCost_function(X, y, theta, dt)
     #   print(curr_cost)
        cost_history.append(curr_cost)
        iters.append(num_iter)
        if (num_iter > 1):
            prev_cost = cost_history[i-1]

            delta_cost = (abs((prev_cost - curr_cost)) * 100 / prev_cost)
            #print(delta_cost, " ", epslon)
            if (delta_cost < epslon):
                print("converged!!! ", num_iter)
                iter = False
                break;

    return cost_history,iters,theta

#partition data
train_size = int(round(len(data) * 0.80))
train = data[:train_size]
test =  data[train_size:]

#1.3 get lasso cost, parameters
cost_history_lasso ,iters_lasso,theta_lasso= gradientDesc_lasso(X, y, train, alpha, theta)
theta_lasso_final = list()
theta_lasso_non_zero = list()
for x in theta_lasso:
    if(abs(x) < 0.005):
        x = 0
        print("not so fast dear")
    else:
        theta_lasso_non_zero.append(x)
    theta_lasso_final.append(x)
print("#4- number of non-zero params from 1.3")
print(theta_lasso_non_zero)

#1.2 get ridge/quad cost,parameters

theta_final = list()
theta_non_zero = list()
cost_history ,iters,theta= gradientDesc(X, y, train, alpha, theta)
for x in theta:
    if(abs(x) < 0.005):
        x = 0
        print("not so fast dear")
    else:
        theta_non_zero.append(x)
    theta_final.append(x)
print("#4- number of non-zero params from 1.2")
print(theta_non_zero)


print("#2::: squared loss on test data")
print(regCost_function(X, y, theta, train))
print(regCost_function(X, y, theta, test))


print("#1::: value of loss function J(theta) vs number of iterations-k on train data")


plt.plot(iters, cost_history)  # Create line plot with yvals against xvals
plt.xlabel('Number of iterations')
plt.ylabel('Value of loss function j(theta)')
plt.show()  # Show the figure




