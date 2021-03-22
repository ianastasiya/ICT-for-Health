# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 17:44:01 2020
@author: isken
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import copy

# To avoid warinings from the terminal for too many openend windows:
plt.rcParams.update({'figure.max_open_warning': 0})
np.random.seed(30)
#matplotlib.rc('text', usetex=True)

class  SolveMinProbl(object):
    
    def __init__(self, ytrain, Xtrain, yval, Xval, ytest, Xtest, mean, std):
        self.Np = ytrain.shape[0]  # Number of patients
        self.Nf = Xtrain.shape[1]  # Number of features
        self.sol = np.zeros((self.Nf, 1), dtype=float)  # Initialize solution

        # Matrices and vectors.
        self.y_train = ytrain.reshape(self.Np, 1)
        self.X_train = Xtrain
        self.y_val = yval.reshape(len(yval), 1)  # Vector with validation data
        self.X_val = Xval  # Matrix with validation data
        self.y_test = ytest.reshape(len(ytest), 1)
        self.X_test = Xtest

        self.err = []  # Mean square error for each iteration on training set
        self.errval = []  # Mean square error for each iteration on validation set
        self.errtest = []  # Mean square error for each iteration on test set
        self.m = mean #mean
        self.s = std #standard deviation

    def plot_w(self, title):

        w = self.sol
        n = np.arange(self.Nf)
        plt.figure()
        plt.plot(n, w.reshape(len(w),), 'd-')
        plt.ylabel(r'$\hat{\mathbf{w}}(f)$')
        plt.xticks(ticks=range(self.Nf), labels=['UPDRS_Motor', 'Jitter_%',
                                                 'Jitter_Abs', 'Jitter_RAP',
                                                 'Jitter_PPQ5', 'Jitter_DDP',
                                                 'Shimmer',
                                                 'Shimmer_dB', 'Shimmer_APQ3',
                                                 'Shimmer_APQ5', 'Shimmer_APQ11',
                                                 'Shimmer_DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE'],
                   rotation='vertical')
        plt.title(title+': Optimum Weight Vector')
        plt.grid(which='both')
        plt.subplots_adjust(bottom=0.25)  # Margin for labels
        
    def plot_data_view(self):
        features=self.UPDRS
        features.plot.hist(bins=50)
        features.plot.scatter(1,2) #Plots the scatter plot, i.e. the data of column i vs the data of column k
        plt.show()    

    def print_result(self):
        print('The optimum weight vector is:')
        print(self.sol, "\n") 
    
    def plot_err(self,title = 'Algorithm', logy=1,logx=0): # calculates the error value of the evaluated data wrt number of iterations 
        err = self.err #plot in color blue
        errval = self.errval
        plt.figure()
        plt.show()
        
        
                # Linear plot
        if (logy == 0) & (logx == 0):
            plt.semilogy(err, color='tab:blue')
            plt.semilogy(errval, color='tab:red')

        # Semilogy plot
        if (logy == 1) & (logx == 0):
            plt.semilogy(err, color='tab:blue')
            plt.semilogy(errval, color='tab:red')

        # Semilogx plot
        if (logy == 0) & (logx == 1):
            plt.semilogy(err, color='tab:blue')
            plt.semilogy(errval, color='tab:red')

        # Loglog plot
        if (logy == 1) & (logx == 1):
            plt.semilogy(err, color='tab:blue')
            plt.semilogy(errval, color='tab:red')

        plt.xlabel('number of iterations')
        plt.ylabel('MSE')
        plt.title(title+': Mean Square Error')
        plt.minorticks_on()
        plt.grid(b=True, which='major')
        plt.legend(['Training set', 'Validation set'])
        plt.savefig("err_"+title.replace(" ", "_")+".pdf")

        
    def graphics(self, title):

        # Reshape vectors for correct plotting.
        ytrain = self.y_train.reshape(len(self.y_train),)*self.s + self.m  # de-standardize
        ytest = self.y_test.reshape(len(self.y_test),)*self.s + self.m
        yhat_train = self.yhat_train.reshape(len(self.yhat_train),)*self.s + self.m  # de-standardize
        yhat_test = self.yhat_test.reshape(len(self.yhat_test),)*self.s + self.m

        # Histogram.
        plt.figure()
        plt.hist(ytrain-yhat_train, bins=50)
        plt.title('y_train - y_hat_train')
        plt.xlabel('Error y_train - y_hat_train')
        plt.ylabel('Instances')
        plt.grid()
        plt.title(title+': Training set')
        plt.xlim([-16, 16])
        plt.ylim([0, 300])
        plt.savefig("h_train_"+title.replace(" ", "_")+".pdf")

        # Histogram.
        plt.figure()
        plt.hist(ytest-yhat_test, bins=50, color='tab:green')
        plt.title('y_test - y_hat_test')
        plt.xlabel('Error y_train - y_hat_train')
        plt.ylabel('Instances')
        plt.grid()
        plt.title(title+': Test set')
        plt.xlim([-16, 16])
        plt.ylim([0, 300])
        plt.savefig("h_test_"+title.replace(" ", "_")+".pdf")

        #  Scatter plot.
        plt.figure()
        plt.scatter(ytrain, yhat_train, marker="2")
        plt.title(title+'y_hat_train vs y_train')
        plt.grid()
        plt.xlabel('y_train')
        plt.ylabel('y_hat_train')
        plt.axis('equal')
        lined = [min(yhat_train), max(yhat_train)]
        plt.plot(lined, lined, color='tab:red')  # Diagonal line
        plt.title(title+': Training set')
        plt.savefig("s_train_"+title.replace(" ", "_")+".pdf")

        #  Scatter plot.
        plt.figure()
        plt.scatter(ytest, yhat_test, marker="2", color='tab:green')
        plt.title(title+': '+'y_hat_test vs y_test')
        plt.grid()
        plt.xlabel('y_test')
        plt.ylabel('y_hat_test')
        plt.axis('equal')
        lined = [min(yhat_train), max(yhat_train)]
        plt.plot(lined, lined, color='tab:red')  # Diagonal line
        plt.title(title+': Test set')
        plt.savefig("s_test_"+title.replace(" ", "_")+".pdf")
        

  
        
class SolveLLS(SolveMinProbl):#class SolveLLS belongs to class SolveMinProb

    def run(self):
        A = self.X_train#Train matrix
        y = self.y_train
        w = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), y)#formula for calculation w 
        self.sol = w
        self.min = np.linalg.norm(np.dot(A, w)-y)**2 #minimization of norm.
        self.yhat_train = np.dot(A, self.sol).reshape(len(y),)
        self.yhat_test = np.dot(self.X_test, self.sol)

        # Errors on standardized vectors.
        # self.err.append((np.linalg.norm(np.dot(A,w)-y)**2)/self.Np)
        # self.errval.append(np.linalg.norm(np.dot(self.X_val,w)-self.y_val)**2/len(self.y_val))
        # self.errtest.append(np.linalg.norm(np.dot(self.X_test,w)-self.y_test)**2/len(self.y_test))

        # Errors on de-standardized vectors.
        self.err.append((np.linalg.norm((np.dot(A, w)*self.s+self.m)-(y*self.s+self.m))**2)/self.Np)
        self.errval.append(np.linalg.norm((np.dot(self.X_val, w)*self.s+self.m) - (self.y_val*self.s+self.m))**2/len(self.y_val))
        self.errtest.append(np.linalg.norm((np.dot(self.X_test, w)*self.s+self.m) - (self.y_test*self.s+self.m))**2/len(self.y_test))

        print('Linear least squares:\n\ttrain_MSE = %.4f\n\ttest_MSE = %.4f\n\tval_MSE = %.4f\n' % (self.err[-1], self.errtest[-1], self.errval[-1]))


class SolveStochasticGradient(SolveMinProbl):#class SolveLLS belongs to class SolveMinProb, Stochastic Gradient Algorithm using Adam method
                                                   
    def run(self, Niteration=1000, gamma=10e-5, eps=1e-3, beta_1 = 0.9, beta_2 = 0.999):
       #Niteration=number of iteration
       #eps(sensibility error)=10^-2=1^-3 always positive: stopping condition, the minimum accepted error
       #gamma(learning coheficient)=10^-5 always positive: the minimum distance to pick a new evaluation point
        t=0                                   
        A = self.X_train                      
        y = self.y_train                           
        t=0                                         
        myu1 = np.zeros((self.Nf,1))
        myu2 = np.zeros((self.Nf,1))
        w = np.random.rand(self.Nf, 1) #start point, random samples from a uniform distribution
        
        for iteration in range(Niteration):
            w2 = copy.deepcopy(w)
            
            for i in range(self.Np):
                t+=1
                grad_i = 2*(np.dot(A[i, :].T, w)-y[i]) * A[i, :].reshape(len(A[i, :]), 1)
                #w = w - gamma*grad_i
                                                                 
                myu1 = beta_1*myu1 + (1-beta_1)*grad_i #updates the moving averages of the gradient
                myu2 = beta_2*myu2 + (1-beta_2)*(grad_i*grad_i) #updates the moving averages of the squared gradient
                mc=myu1/(1-beta_1**t) #calculates the bias-corrected estimates
                sc=myu2/(1-beta_2**t) #calculates the bias-corrected estimates
                w= w - (gamma*mc)/(np.sqrt(sc)+eps) #updates the parameters
            
            if np.linalg.norm(w2-w) < eps:
              print("Stochastic gradient descent has stopped after %d iterations, MSE = %4f" % (iteration, self.err[-1]))
              break

            # Errors on standardized vectors.
#            self.errtrain.append(np.linalg.norm(np.dot(A, w)-y)**2/self.Np) # Average of errorTrain = minTrain/nTrain
#            self.errval.append(np.linalg.norm(np.dot(self.X_val, w)-self.y_val)**2/len(self.y_val)) # Average of errorVal = minVal/nVal
#            self.errtest.append(np.linalg.norm(np.dot(self.X_test, w)-self.y_test)**2/len(self.y_test)) # Average of errorTest = minTest/nTest

    # Errors on de-standardized vectors.
            self.err.append((np.linalg.norm((np.dot(A, w)*self.s+self.m)-(y*self.s+self.m))**2)/self.Np)
            self.errval.append(np.linalg.norm((np.dot(self.X_val, w)*self.s+self.m) - (self.y_val*self.s+self.m))**2/len(self.y_val))
            self.errtest.append(np.linalg.norm((np.dot(self.X_test, w)*self.s+self.m) - (self.y_test*self.s+self.m))**2/len(self.y_test))
    



        self.sol = w
        self.min = self.err[-1] #the minimization=the errorTrain because we already minimized it with the gradient
        self.yhat_train = np.dot(A, self.sol).reshape(len(y),) # Ytrain=W.Xtrain in order to comparate Ytrain with Ytest
        self.yhat_test = np.dot(self.X_test, self.sol) # Ytest=Xtest.W in order to comparate Ytrain with Ytest
        
        # print the MSE=mean square error of the Train, Test and Validation
        print('\nStochastic gradient descent:\nthe mean square error of the Train, Test and Validation:\n\ttrain_MSE = %.4f\n\ttest_MSE = %.4f\n\tval_MSE = %.4f\n' % (self.err[-1], self.errtest[-1], self.errval[-1])) 
        #Return the last value of the error(ours last average)    

class SolveConjugateGradient(SolveMinProbl):

    def run(self):
        A = self.X_train
        y = self.y_train
        w = np.zeros((self.Nf, 1), dtype=float)
        Q = np.dot(A.T, A)
        b = np.dot(A.T, y)
        d = b
        g = -b

        for it in range(self.Nf):  # Iterations on number of features
            alpha = -((np.dot(d.T, g))/(np.dot(np.dot(d.T, Q), d)))
            w = w + alpha*d
            # g = np.dot(Q,w) - b
            g = g + alpha*(np.dot(Q, d))
            beta = np.dot(np.dot(g.T, Q), d)/np.dot(np.dot(d.T, Q), d)
            d = -g + beta*d
            # Errors on standardized vectors.
            # self.err.append((np.linalg.norm(np.dot(A,w)-y)**2)/self.Np)
            # self.errval.append(np.linalg.norm(np.dot(self.X_val,w)-self.y_val)**2/len(self.y_val))
            # self.errtest.append(np.linalg.norm(np.dot(self.X_test,w)-self.y_test)**2/len(self.y_test))

            # Errors on de-standardized vectors.
            self.err.append((np.linalg.norm((np.dot(A, w)*self.s+self.m)-(y*self.s+self.m))**2)/self.Np)
            self.errval.append(np.linalg.norm((np.dot(self.X_val, w)*self.s + self.m)-(self.y_val*self.s+self.m))**2/len(self.y_val))
            self.errtest.append(np.linalg.norm((np.dot(self.X_test, w)*self.s + self.m)-(self.y_test*self.s+self.m))**2/len(self.y_test))

        self.sol = w
        self.min = self.err[-1]
        self.yhat_train = np.dot(A, self.sol).reshape(len(y),)
        self.yhat_test = np.dot(self.X_test, self.sol)
        print('Conjugate gradient method:\n\ttrain_MSE = %.4f\n\ttest_MSE = %.4f\n\tval_MSE = %.4f\n' % (self.err[-1], self.errtest[-1], self.errval[-1]))
        print("Conjugate gradient has stopped after %d iterations, MSE = %4f" % (it+1, self.err[-1]))

class SolveRidge(SolveMinProbl):
    
    def run(self, Lambda=np.linspace(0.001, 80, num=300)):

        self.lambda_range = Lambda
        self.errvalid = []
        it = 0
        for L in Lambda:
            A = self.X_train
            y = self.y_train
            w = np.random.rand(self.Nf, 1)
            I = np.eye(self.Nf)
            w = np.dot(np.dot(np.linalg.inv((np.dot(A.T, A) + L*I)), A.T), y)

            # Errors on standardized vectors.
            # self.err.append((np.linalg.norm(np.dot(A,w)-y)**2)/self.Np)
            self.errvalid.append(np.linalg.norm(np.dot(self.X_val, w)-self.y_val)**2/len(self.y_val))
            # self.errtest.append(np.linalg.norm(np.dot(self.X_test,w)-self.y_test)**2/len(self.y_test))

            # Errors on de-standardized vectors.
            self.err.append((np.linalg.norm((np.dot(A, w)*self.s+self.m)-(y*self.s+self.m))**2)/self.Np)
            self.errtest.append(np.linalg.norm((np.dot(self.X_test, w)*self.s + self.m)-(self.y_test*self.s+self.m))**2/len(self.y_test))

            self.min = min(self.errvalid)

            if self.errvalid[-1] <= self.min:
                self.sol = w
                it_best = it
                lambda_best = L
                self.yhat_train = np.dot(A, self.sol).reshape(len(y),)
                self.yhat_test = np.dot(self.X_test, self.sol)
            it += 1
        print('Ridge regression:\n\ttrain_MSE = %.4f\n\ttest_MSE = %.4f\n' %
              (self.err[it_best], self.errtest[it_best]))
        print('\tLambda = %f' % lambda_best)

    def plotRidgeError(self, title='Algorithm'):

        plt.figure()
        plt.plot(self.lambda_range, self.err, color='tab:blue')
        plt.plot(self.lambda_range, self.errtest, color='tab:red', linestyle=':')
        plt.xlabel(r'$\lambda$')
        plt.ylabel('Mean Square Error')
        plt.title('Ridge regression: mean square error')
        plt.grid()
        plt.legend(['Training set', 'Validation set'])
        plt.savefig("err_"+title.replace(" ", "_")+".pdf")



if __name__ == '__main__':
     #F0=7 ##shimmer      
    F0 = 1  # updrs column
    data = pd.read_csv("parkinsons_updrs.csv")  # Import CSV into a dataframe
    data = data.drop(columns=['subject#', 'age', 'sex', 'test_time'])  # Drop the first columns
    data = data.sample(frac=1).reset_index(drop=True)  # Shuffle rows and reset index
 
    # Submatrices: training, validation and test
    data_train = data[0:math.ceil(data.shape[0]/2)-1]
    data_val = data[math.floor(data.shape[0]/2):math.floor(3/4*data.shape[0])]
    data_test = data[math.floor(3/4*data.shape[0]):data.shape[0]]
    

    
    
    # Data normalization.
    data_train_norm = copy.deepcopy(data_train)  # To preserve original data
    data_val_norm = copy.deepcopy(data_val)  # To preserve original data
    data_test_norm = copy.deepcopy(data_test)  # To preserve original data

    for i in range(data_train.shape[1]):#for i in columns of data_train
        mean = np.mean(data_train.iloc[:, i])  # Calculate mean for data_train
        data_train_norm.iloc[:, i] -= mean
        data_val_norm.iloc[:, i] -= mean
        data_test_norm.iloc[:, i] -= mean
        std = np.std(data_train.iloc[:, i])  # Calculate standard deviation for data_train
        data_train_norm.iloc[:, i] /= std
        data_val_norm.iloc[:, i] /= std
        data_test_norm.iloc[:, i] /= std
        
    # Mean and standard deviation in order to de-standardize data for the plots.
    m = np.mean(data_train.iloc[:, F0]) #Mean of the column motor_UPDRS
    s = np.std(data_train.iloc[:, F0]) #Standard deviation of the column motor_UPDRS

    #The regressand will be (Y=total_UPDRS) and regressors(X=all the other features including motor UPDRS)
    y_train = data_train_norm.iloc[:, F0]  # Just column total_UPDRS
    y_test = data_test_norm.iloc[:, F0]  # Just column total_UPDRS
    y_val = data_val_norm.iloc[:, F0]  # Just column total_UPDRS
    
    X_train = data_train_norm.drop(columns='total_UPDRS')  # Remove total_UPDRS and let the others features
    X_test = data_test_norm.drop(columns='total_UPDRS')  # Remove total_UPDRS and let the others features
    X_val = data_val_norm.drop(columns='total_UPDRS')  # Remove total_UPDRS and let the others features
  
    # Class initializations.
    lls = SolveLLS(y_train.values, X_train.values, y_val.values, X_val.values, y_test.values, X_test.values, m, s)

    scg = SolveConjugateGradient(y_train.values, X_train.values, y_val.values, X_val.values, y_test.values, X_test.values, m, s)
  
    ssg = SolveStochasticGradient(y_train.values, X_train.values, y_val.values, X_val.values, y_test.values, X_test.values, m, s)

    ridge = SolveRidge(y_train.values, X_train.values, y_val.values, X_val.values, y_test.values, X_test.values, m, s)

#    # Linear least squares.
#    lls.run()
#    lls.print_result()
#    lls.plot_w('Linear Least Squares')
#    lls.graphics('Linear Least Squares')
###
###     # Conjugate gradient descent.
#    scg.run()
#    scg.print_result()
##    scg.plot_w('Conjugate Gradient Method')
#    scg.plot_err('Conjugate Gradient Method')
##    scg.graphics('Conjugate Gradient Method')
###
###    # Stochastic gradient descent.
#    ssg.run()
#    ssg.print_result()
##    ssg.plot_w('Stochastic Gradient Descent')
##    ssg.plot_err('Stochastic Gradient Descent')
#    ssg.graphics('Stochastic Gradient Descent')

    # Ridge regression.
    ridge.run()
    ridge.print_result()
    ridge.plot_w('Ridge Regression')
    ridge.plotRidgeError('Ridge Regression')
    ridge.graphics('Ridge Regression')

print("\nEND OF THE CALCULATIONS\n")  