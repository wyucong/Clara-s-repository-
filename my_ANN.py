#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 18:29:24 2019

@author: yucongweng
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
import numpy as np

class ANN(object):
   
    def __init__(self, hidden_layer_sizes):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.losses = []
        self.scores = []
        self.W2 = None
        self.b2 = None
        self.W1 = None
        self.b1 = None

        
    def derivative_w2(self, Z, T, Y):
        
        return Z.T.dot(Y - T)

    def derivative_b2(self, T, Y):
        return np.sum(Y-T, axis = 0)

    def derivative_w1(self, X, Z, T, Y, activation):
        if activation == 'relu':
            return X.T.dot( ( ( Y-T ).dot(self.W2.T) * (Z > 0) ) ) # for relu
        if activation == 'sigmoid':
            return X.T.dot( ( ( Y-T ).dot(self.W2.T) * ( Z*(1 - Z) ) ) ) # for sigmoid

    def derivative_b1(self, T, Y,  Z, activation):
        if activation == 'relu':
            return (( Y-T ).dot(self.W2.T) * (Z > 0)).sum(axis=0) # for relu
        if activation == 'sigmoid':
     
            return np.sum((Y-T).dot(self.W2.T) * Z * (1-Z), axis = 0) # for sigmoid
    
    def forward(self, X, activation):
        if activation == 'relu':
            Z = X.dot(self.W1) + self.b1
            Z[Z<0] = 0
        if activation == 'sigmoid':
            Z = 1 / (1 + np.exp(-( X.dot(self.W1) + self.b1 )))
        A = Z.dot(self.W2) + self.b2
        expA = np.exp(A)
        Y = expA / expA.sum(axis = 1, keepdims = True)
        return Y, Z
    
        

    def score(self, Y, T):
        return np.mean(Y == T)
    
    def predict(self, p_y):
        return np.argmax(p_y, axis = 1)
    
    def cross_entropy(self, T, pY):
        return -np.mean(T*np.log(pY))
    
    def _y2indicator(self, y):
        K = len(set(y))
        result = np.zeros((len(y), K))
        for i in range(len(y)):
            result[i, y[i]] = 1
        return result
    
    def fit_adam(self, X, Y, activation='relu',  learning_rate=1e-3, beta1 = 0.95, beta2 = 0.999, eps = 1e-8, reg=0.05, epochs=1000, batch_sz=None, print_period = 100):
        Y_ind = self._y2indicator(Y)
        
        K = len(set(Y))
        M = self.hidden_layer_sizes 
        D = X.shape[1]
        self.W1 = np.random.randn(D, M)
        self.b1 = np.random.randn(M)
        self.W2 = np.random.randn(M, K)
        self.b2 = np.random.randn(K)
        
        N = X.shape[0]
        batch_sz = batch_sz
        n_batches = N // batch_sz
        
        mW1 = 0
        mb1 = 0
        mW2 = 0
        mb2 = 0
        
        vW1 = 0
        vb1 = 0
        vW2 = 0
        vb2 = 0
        
        t = 1
        for i in range(epochs):
            for j in range(n_batches):
                Xbatch = X[j*batch_sz:(j*batch_sz + batch_sz),]
                ybatch = Y_ind[j*batch_sz:(j*batch_sz + batch_sz),]
                pYbatch, hidden = self.forward(Xbatch,  activation)
 
        
                # gradient
                gW2 = self.derivative_w2(hidden, ybatch, pYbatch) + reg * self.W2 
                gb2 = self.derivative_b2(ybatch, pYbatch) + reg * self.b2
                gW1 = self.derivative_w1(Xbatch, hidden, ybatch, pYbatch,  activation)+ reg * self.W1 
                gb1 = self.derivative_b1(ybatch, pYbatch,  hidden, activation) + reg * self.b1 
        
                # update m
                mW1 = beta1 * mW1 + (1 - beta1) * gW1
                mb1 = beta1 * mb1 + (1 - beta1) * gb1
                mW2 = beta1 * mW2 + (1 - beta1) * gW2
                mb2 = beta1 * mb2 + (1 - beta1) * gb2
                
                # update v
                vW1 = beta2 * vW1 + (1 - beta2) * gW1 * gW1
                vb1 = beta2 * vb1 + (1 - beta2) * gb1 * gb1
                vW2 = beta2 * vW2 + (1 - beta2) * gW2 * gW2
                vb2 = beta2 * vb2 + (1 - beta2) * gb2 * gb2
        
                #bias correction
                correction1 = 1 - beta1**t
                hat_mW1 = mW1 / correction1
                hat_mb1 = mb1 / correction1
                hat_mW2 = mW2 / correction1
                hat_mb2 = mb2 / correction1
        
                correction2 = 1 - beta2**t
                hat_vW1 = vW1 / correction2
                hat_vb1 = vb1 / correction2
                hat_vW2 = vW2 / correction2
                hat_vb2 = vb2 / correction2
        
                t += 1
                
                # combine the two
                self.W1 = self.W1 - learning_rate * hat_mW1 / np.sqrt(hat_vW1 + eps)
                self.b1 = self.b1 - learning_rate * hat_mb1 / np.sqrt(hat_vb1 + eps)
                self.W2 = self.W2 - learning_rate * hat_mW2 / np.sqrt(hat_vW2 + eps)
                self.b2 = self.b2 - learning_rate * hat_mb2 / np.sqrt(hat_vb2 + eps)
        
                if j % print_period == 0:
                    pY, _ = self.forward(X,  activation)
                    l = self.cross_entropy(Y_ind, pY)
                    self.losses.append(l)
                    print('cost at iteration i = %d, j = %d: %.6f' % (i,j,l))
            
                    a = self.score(Y, self.predict(pY))
                    self.scores.append(a)
                    print('test clasification rate: ', a)
       
    
    def fit_momendum(self, X, Y, activation='relu',  learning_rate=1e-3, mu=0.0, reg=0, epochs=1000, batch_sz=None, print_period = 100):
        Y_ind = self._y2indicator(Y)
        
        K = len(set(Y))
        M = self.hidden_layer_sizes 
        D = X.shape[1]
        self.W1 = np.random.randn(D, M)
        self.b1 = np.random.randn(M)
        self.W2 = np.random.randn(M, K)
        self.b2 = np.random.randn(K)
        
        N = X.shape[0]
        batch_sz = batch_sz
        n_batches = N // batch_sz
        
        dW1 = 0
        db1 = 0
        dW2 = 0
        db2 = 0
        
       
        for i in range(epochs):
            for j in range(n_batches):
                Xbatch = X[j*batch_sz:(j*batch_sz + batch_sz),]
                ybatch = Y_ind[j*batch_sz:(j*batch_sz + batch_sz),]
                pYbatch, hidden = self.forward(Xbatch,  activation)
 
        
                # gradient
                gW2 = self.derivative_w2(hidden, ybatch, pYbatch) + reg * self.W2 
                gb2 = self.derivative_b2(ybatch, pYbatch) + reg * self.b2
                gW1 = self.derivative_w1(Xbatch, hidden, ybatch, pYbatch,  activation)+ reg * self.W1 
                gb1 = self.derivative_b1(ybatch, pYbatch,  hidden, activation) + reg * self.b1 
        
                # update velocities
                dW2 = mu*dW2 - learning_rate * gW2
                db2 = mu*db2 - learning_rate * gb2
                dW1 = mu*dW1 - learning_rate * gW1
                db1 = mu*db1 - learning_rate * gb1
        
            # updates
                self.W2 += dW2
                self.b2 += db2
                self.W1 += dW1
                self.b1 += db1
        
                if j % print_period == 0:
                    pY, _ = self.forward(X,  activation)
                    l = self.cross_entropy(Y_ind, pY)
                    self.losses.append(l)
                    print('cost at iteration i = %d, j = %d: %.6f' % (i,j,l))
            
                    a = self.score(Y, self.predict(pY))
                    self.scores.append(a)
                    print('test clasification rate: ', a)
       