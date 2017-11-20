#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 14:33:13 2017

@author: eilers
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.optimize as op
import pickle
from astropy.table import Column
from astropy.table import Table
from sklearn.decomposition import PCA
from itertools import product



# -------------------------------------------------------------------------------
# kernel
# -------------------------------------------------------------------------------

# radius basis function
def kernelRBF(Z, rbf, band): 
    B = B_matrix(Z)
    kernel = rbf * np.exp(band * B) 
    return kernel


# dimensionless log kernel
def B_matrix(Z):
    N = Z.shape[0]
    B = np.zeros((N, N))
    for entry in list(product(range(N), repeat=2)):
        i, j = entry
        # delta = Z[i, :] - Z[j,:]
        # -0.5 * delta.T @ delta
        B[i, j] = -0.5 * np.dot((Z[i, :] - Z[j, :]).T, (Z[i, :] - Z[j, :]))
    return B

# -------------------------------------------------------------------------------
# optimization of latent variables
# -------------------------------------------------------------------------------

def lnL_Z(pars, X, Y, hyper_params, Z_initial, X_var, Y_var):  
    
    N = Z_initial.shape[0]
    Q = Z_initial.shape[1]
    D = X.shape[1]
    L = Y.shape[1]
    Z = np.reshape(pars, (N, Q)) 
    theta_rbf, theta_band, gamma_rbf, gamma_band = hyper_params    
    kernel1 = kernelRBF(Z, theta_rbf, theta_band)
    kernel2 = kernelRBF(Z, gamma_rbf, gamma_band)
    
    Lx, Ly, gradLx, gradLy = 0., 0., 0., 0.
    for d in range(D):
        K1C = kernel1 + np.diag(X_var[:, d])
        Lx += LxOrLy(K1C, X[:, d])   
        gradLx += dLdZ(X[:, d], Z, K1C, theta_rbf, theta_band)        
        
    for l in range(L):
        K2C = kernel2 + np.diag(Y_var[:, l])
        Ly += LxOrLy(K2C, Y[:, l]) 
        gradLy += dLdZ(Y[:, l], Z, K2C, gamma_rbf, gamma_band)    
        
    #for n in range(N):
    #    Lz += -0.5 * np.log(2*np.pi) -0.5 * np.dot(Z[n, :].T, Z[n, :])
    #    dlnpdZ += -1. * Z[n, :]
        
    Lz = -0.5 * np.log(2.*np.pi) - 0.5 * np.sum(Z**2)
    dlnpdZ = -Z 
    L = Lx + Ly + Lz   
    gradL = gradLx + gradLy + dlnpdZ      
    gradL = np.reshape(gradL, (N * Q, ))   # reshape gradL back into 1D array   
    print -2.*Lx, -2.*Ly, -2.*Lz
    
    return -2.*L, -2.*gradL#, -2.*Lx, -2.*np.reshape(gradLx, (N * Q, )), -2.*Ly, -2.*np.reshape(gradLy, (N * Q, )), -2.*Lz, -2.*np.reshape(dlnpdZ, (N * Q, ))


def LxOrLy(K, data):  
    L_term1 = -0.5 * data.shape[0] * np.log(2.*np.pi)   # * data.shape[1], if more than one dimension! now: implemented as sum in lnL_Z!
    L_term2 = -0.5 * np.linalg.slogdet(K)[1]            # * data.shape[1], if more than one dimension! now: implemented as sum in lnL_Z!
    L_term3 = -0.5 * np.matrix.trace(np.dot(np.linalg.inv(K), np.dot(data, data.T)))
    return L_term1 + L_term2 + L_term3

# -------------------------------------------------------------------------------
# derivatives 
# -------------------------------------------------------------------------------

# derivative of the kernel with respect to the latent variables Z
def dKdZ(Z, rbf, band):
    kernel = kernelRBF(Z, rbf, band)
    A = A_matrix(Z)
    grad_dKdZ = band * kernel[:, :, None, None] * A
    return grad_dKdZ


def A_matrix(Z): 
    N, Q = Z.shape[0], Z.shape[1]
    A = np.zeros((N, N, N, Q))
    for entry in list(product(range(N), repeat=3)):
        i, j, l = entry
        if l == j:
            A[i, j, l, :] += Z[i, :] - Z[l, :]
        if l == i:
            A[i, j, l, :] += Z[j, :] - Z[l, :]
    return A 


def dLdK(K, data):   
    inv_K = np.linalg.inv(K)
    grad_dLdK = -0.5 * inv_K + 0.5 * (np.dot(np.dot(inv_K, np.dot(data, data.T)), inv_K)) # first term * data.shape[1], if more than one dimension!
    return grad_dLdK                     # shape: N x N 
    


def dLdZ(data, Z, KC, rbf, band):       
    grad_dKdZ = dKdZ(Z, rbf, band)
    grad_dLdK = dLdK(KC, data)    
    gradL = np.sum(grad_dLdK[:, :, None, None] * grad_dKdZ, axis = (0, 1))
    return gradL


# -------------------------------------------------------------------------------
# optimization of hyper parameters
# -------------------------------------------------------------------------------


def lnL_h(pars, X, Y, Z, Z_initial, X_var, Y_var):
    
    # hyper parameters shouldn't be negative
    if all(i >= 0 for i in pars):    
        N = Z_initial.shape[0]
        Q = Z_initial.shape[1]
        D = X.shape[1]
        L = Y.shape[1]    
        Z = np.reshape(Z, (N, Q))  
        theta_rbf, theta_band, gamma_rbf, gamma_band = pars 
        
        kernel1 = kernelRBF(Z, theta_rbf, theta_band)
        kernel2 = kernelRBF(Z, gamma_rbf, gamma_band)
        
        Lx, Ly, gradLx, gradLy = 0., 0., 0., 0.
        for d in range(D):
            K1C = kernel1 + np.diag(X_var[:, d])
            Lx += LxOrLy(K1C, X[:, d])   
            gradLx += dLdhyper(X[:, d], Z, theta_band, theta_rbf, kernel1, K1C) 
            
        for l in range(L):
            K2C = kernel2 + np.diag(Y_var[:, l])
            Ly += LxOrLy(K2C, Y[:, l]) 
            gradLy += dLdhyper(Y[:, l], Z, gamma_band, gamma_rbf, kernel2, K2C) 
         
        Lz = -0.5 * np.log(2.*np.pi) - 0.5 * np.sum(Z**2)                      
        L = Lx + Ly + Lz
        gradL = np.hstack((gradLx, gradLy))
        
        print -2.*Lx, -2.*Ly 
        return -2.*L, -2.*gradL
    else:
        print('hyper parameters negative!') 
        return 1e12, 1e12 * np.ones_like(pars) # hack! check again!


def dLdhyper(data, Z, band, rbf, K, KC):  
    dKdrbf = 1./rbf * K    
    dKdband = K * B_matrix(Z)
    
    dLdrbf = np.sum(dLdK(KC, data) * dKdrbf)        
    dLdband = np.sum(dLdK(KC, data) * dKdband)    
    
    return np.array([dLdrbf, dLdband])


# -------------------------------------------------------------------------------
# prediction
# -------------------------------------------------------------------------------

def mean_var(Z, Zj, data, data_var, K, rbf, band):
    N = data.shape[0]
    B = np.zeros((N, ))
    for i in range(N):
        B[i] = -0.5 * np.dot((Z[i, :] - Zj).T, (Z[i, :] - Zj))   
    k_Z_zj = rbf * np.exp(band * B)
    
    # prediction for test object: loop over d is in previous function
    if data.ndim == 1:
        inv_K = np.linalg.inv(K + np.diag(data_var))
        mean = np.dot(data.T, np.dot(inv_K, k_Z_zj))
        var = rbf - np.dot(k_Z_zj.T, np.dot(inv_K, k_Z_zj))
        return mean, var, k_Z_zj, inv_K
    
    # prediction for training objects
    else:
        D = data.shape[1]
        mean_j = []
        var_j = []
        for d in range(D):
            inv_K = np.linalg.inv(K + np.diag(data_var[:, d]))
            mean_j.append(np.dot(data[:, d].T, np.dot(inv_K, k_Z_zj)))
            var_j.append(rbf - np.dot(k_Z_zj.T, np.dot(inv_K, k_Z_zj)))        
        return np.array(mean_j), np.array(var_j), k_Z_zj
    


def lnL(pars, X_new_j, X_new_var_j, Z, data, data_var, K, rbf, band):    
    Zj = pars
    D = X_new_j.shape[0]
    
    L = 0.
    gradL = 0.
    for d in range(D):
        mean, var, k_Z_zj, inv_K = mean_var(Z, Zj, data[:, d], data_var[:, d], K, rbf, band)
        #assert var > 0.

        L += -0.5 * np.dot((X_new_j[d] - mean).T, (X_new_j[d] - mean)) / \
                          (var + X_new_var_j[d]) - 0.5 * np.log(var + X_new_var_j[d]) 

        dLdmu, dLdsigma2 = dLdmusigma2(X_new_j[d], mean, (var + X_new_var_j[d]))
        dmudZ, dsigma2dZ = dmusigma2dZ(data[:, d], inv_K, Z, Zj, k_Z_zj, band)
        
        gradL += np.dot(dLdmu, dmudZ) + np.dot(dLdsigma2, dsigma2dZ)
    
    return -2.*L, -2.*gradL


def dLdmusigma2(X_new_j, mean, var):
    dLdmu = (X_new_j - mean) / var
    dLdsigma2 = -0.5 / var + 0.5 * np.dot((X_new_j - mean).T, (X_new_j - mean)) / var**2.
    return dLdmu, dLdsigma2


def dmusigma2dZ(data, inv_K, Z, Zj, k_Z_zj, band):
    term1 = np.dot(data.T, inv_K)
    term2 = k_Z_zj[:, None] * band * (Z - Zj)
    dmudZ = np.dot(term1, term2)    
    
    term3 = -2. * np.dot(k_Z_zj , inv_K)
    dsigma2dZ = np.dot(term3, term2)
    
    return dmudZ, dsigma2dZ


def predictX(N_new, X_new, X_new_var, X, X_var, Z_final, K, hyper_params):
    
    Q = Z_final.shape[1]
    theta_rbf, theta_band, gamma_rbf, gamma_band = hyper_params
    Z_new = np.zeros((N_new, Q))
    # first guess: 
    x0 = np.mean(Z_final, axis = 0)
    
    for j in range(N_new):
        res = op.minimize(lnL, x0 = x0, args = (X_new[j, :], X_new_var[j, :], Z_final, X, X_var, K, theta_rbf, theta_band), method = 'L-BFGS-B', jac = True, 
                       options={'gtol':1e-12, 'ftol':1e-12})   
        Z_new[j, :] = res.x
        print('success: {}'.format(res.success))
    
    return Z_new


# -------------------------------------------------------------------------------
# data
# -------------------------------------------------------------------------------

def make_label_input(labels, training_labels):
    tr_label_input = np.array([training_labels[x] for x in labels]).T
    #tr_ivar_input = 1./((np.array([training_labels[x+'_ERR'] for x in labels]).T)**2)
    tr_var_input = (np.array([training_labels[x+'_ERR'] for x in labels]).T)**2
    for x in range(tr_label_input.shape[1]):
        bad = np.logical_or(tr_label_input[:, x] < -100., tr_label_input[:, x] > 9000.) # magic
        tr_label_input[bad, x] = np.median(tr_label_input[:, x])
        #tr_ivar_input[bad, x] = 0.
        tr_var_input[bad, x] = 1e8
    # remove one outlier in T_eff and [N/Fe]!
    bad = tr_label_input[:, 0] > 5200.
    tr_label_input[bad, 0] = np.median(tr_label_input[:, 0])
    #tr_ivar_input[bad, 0] = 0. 
    tr_var_input[bad, 0] = 1e8 
    #bad = tr_label_input[:, 5] < -0.6
    #tr_label_input[bad, 5] = np.median(tr_label_input[:, 5])
    #tr_ivar_input[bad, 5] = 0.     
    return tr_label_input, tr_var_input


# scale and pivot labels
def get_pivots_and_scales(label_vals):  
    qs = np.percentile(label_vals, (2.5, 50, 97.5), axis=0)
    pivots = qs[1]
    scales = (qs[2] - qs[0])/4.    
    return pivots, scales


def PCAInitial(X, Q):
    pca = PCA(Q)
    Z_initial = pca.fit_transform(X)
    return Z_initial