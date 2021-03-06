#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 14:33:13 2017

@author: eilers
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import pickle
from astropy.table import Column
from astropy.table import Table
from sklearn.decomposition import PCA
from itertools import product
from scipy.linalg import cho_solve, cho_factor
from choldate import cholupdate, choldowndate
import os.path
import subprocess
from astropy.io import fits


nervous = False

# -------------------------------------------------------------------------------
# kernel
# -------------------------------------------------------------------------------

# radius basis function
def kernelRBF(Z, rbf, band): 
    B = B_matrix(Z)
    kernel = rbf * np.exp(band * B) 
    return kernel


# dimensionless log kernel
def B_matrix_old(Z):
    N = Z.shape[0]
    B = np.zeros((N, N))
    for entry in list(product(range(N), repeat=2)):
        i, j = entry
        # delta = Z[i, :] - Z[j, :]
        # -0.5 * delta.T @ delta
        B[i, j] = -0.5 * np.dot((Z[i, :] - Z[j, :]).T, (Z[i, :] - Z[j, :]))
    return B

# only marginally faster than B_matrix_old (uses more memory)
def B_matrix(Z):
    return -0.5 * np.sum((Z[None, :, :] - Z[:, None, :]) ** 2, axis=2)

# faster than B_matrix() but gives oddly different results end-to-end.
def B_matrix_new(Z):
    ZZ = np.sum(Z * Z, axis=1)
    return -0.5 * ZZ[:, None] + np.dot(Z, Z.T) - 0.5 * ZZ[None, :]
    
# -------------------------------------------------------------------------------
# optimization of latent variables
# -------------------------------------------------------------------------------

def cygnet_likelihood_d_worker(task):
    obj, d = task
    good_stars = obj.X_mask[:, d]
    thiskernel = obj.kernel1[good_stars, :][:, good_stars]
    K1C = thiskernel + np.diag(obj.X_var[good_stars, d])
    thisfactor = cho_factor(K1C, overwrite_a = True)
    thislogdet = 2. * np.sum(np.log(np.diag(thisfactor[0])))
    Lx = LxOrLy(thislogdet, thisfactor, obj.X[good_stars, d])
    gradLx = np.zeros_like(obj.Z)
    gradLx[good_stars, :] = dLdZ(obj.X[good_stars, d], obj.Z[good_stars, :], thisfactor, thiskernel, obj.theta_band)        
    return Lx, gradLx

def cygnet_likelihood_l_worker(task):
    obj, l = task
    
    good_stars = obj.Y_mask[:, l]
    thiskernel = obj.kernel2[good_stars, :][:, good_stars]
    K2C = thiskernel + np.diag(obj.Y_var[good_stars, l])
    thisfactor = cho_factor(K2C, overwrite_a = True)
    thislogdet = 2. * np.sum(np.log(np.diag(thisfactor[0])))
    Ly = LxOrLy(thislogdet, thisfactor, obj.Y[good_stars, l])
    gradLy = np.zeros_like(obj.Z)
    gradLy[good_stars, :] = dLdZ(obj.X[good_stars, l], obj.Z[good_stars, :], thisfactor, thiskernel, obj.gamma_band)        
    return Ly, gradLy

class CygnetLikelihood():
    
    def __init__(self, X, Y, Q, hyper_params, X_var, Y_var, X_mask = None, Y_mask = None):
        '''
        X: pixel data
        Y: label data
        '''
        # change X and Y!
        self.X = X.copy()
        self.Y = Y.copy()
        self.X_var = X_var.copy()
        self.Y_var = Y_var.copy()
        assert self.X.shape == self.X_var.shape
        assert self.Y.shape == self.Y_var.shape        
        self.N, self.D = self.X.shape
        N, self.L = self.Y.shape
        assert N == self.N
        self.Q = Q
        self.theta_rbf, self.theta_band, self.gamma_rbf, self.gamma_band = hyper_params
        if X_mask is None:
            self.X_mask = np.ones_like(X).astype(bool)
        else:
            self.X_mask = X_mask.copy()
        if Y_mask is None:
            self.Y_mask = np.ones_like(Y).astype(bool)    
        else:
            self.Y_mask = Y_mask.copy()
        
        # container to hold the summed likelihood value, gradients
        self._L = 0.
        self._gradL = np.zeros_like(self.Z)
        
    def update_pars(self, pars):
        
        self.Z = np.reshape(pars, (self.N, self.Q))
        self.kernel1 = kernelRBF(self.Z, self.theta_rbf, self.theta_band)
        self.kernel2 = kernelRBF(self.Z, self.gamma_rbf, self.gamma_band)

    def __call__(self, pars, pool):
        self.update_pars(pars)        
        
        tasks = [(self, d) for d in range(self.D)]
        Lx = 0.
        gradLx = np.zeros_like(self.Z)
        for result in pool.map(cygnet_likelihood_d_worker, tasks):
            Lx += result[0]
            gradLx += result[1]
        
        # reset containers
        cygnet_likelihood._L = 0.    
        cygnet_likelihood._gradL = np.zeros_like(cygnet_likelihood.Z)
        for _ in pool.map(cygnet_likelihood, zip(range(L), ['l']*L)):
            pass
        Ly = cygnet_likelihood._L
        gradLy = cygnet_likelihood._gradL
        
        Lz = -0.5 * np.sum(cygnet_likelihood.Z**2)
        dlnpdZ = -cygnet_likelihood.Z 
        L = Lx + Ly + Lz   
        gradL = gradLx + gradLy + dlnpdZ      
        gradL = np.reshape(gradL, (cygnet_likelihood.N * cygnet_likelihood.Q, ))   # reshape gradL back into 1D array   
        print(-2.*Lx, -2.*Ly, -2.*Lz)
        
        return -2.*L, -2.*gradL
        
        
        if l_or_d == 'd':
            return self.lnLd(index)
        elif l_or_d == 'l':
            return self.lnLl(index)
    
    def lnLl(self, l):
        good_stars = self.Y_mask[:, l]
        thiskernel = self.kernel2[good_stars, :][:, good_stars]
        K2C = thiskernel + np.diag(self.Y_var[good_stars, l])
        thisfactor = cho_factor(K2C, overwrite_a = True)
        thislogdet = 2. * np.sum(np.log(np.diag(thisfactor[0])))
        Ly = LxOrLy(thislogdet, thisfactor, self.Y[good_stars, l])
        gradLy = np.zeros_like(self.Z)
        gradLy[good_stars, :] = dLdZ(self.X[good_stars, l], self.Z[good_stars, :], thisfactor, thiskernel, self.gamma_band)        
        return Ly, gradLy
    
    def callback(self, result):
        _L, _gradL = result
        self._L += _L
        self._gradL += _gradL

def lnL_Z(pars, X, Y, Q, hyper_params, pool, X_var, Y_var, X_mask = None, Y_mask = None):  
        
    cygnet_likelihood = CygnetLikelihood(pars, X, Y, Q, hyper_params, X_var, Y_var, X_mask, Y_mask)
    D = cygnet_likelihood.D    
    L = cygnet_likelihood.L    
    
    for _ in pool.map(cygnet_likelihood, zip(range(D), ['d']*D)):
        pass
    Lx = cygnet_likelihood._L
    gradLx = cygnet_likelihood._gradL
    
    # reset containers
    cygnet_likelihood._L = 0.    
    cygnet_likelihood._gradL = np.zeros_like(cygnet_likelihood.Z)
    for _ in pool.map(cygnet_likelihood, zip(range(L), ['l']*L)):
        pass
    Ly = cygnet_likelihood._L
    gradLy = cygnet_likelihood._gradL
    
    Lz = -0.5 * np.sum(cygnet_likelihood.Z**2)
    dlnpdZ = -cygnet_likelihood.Z 
    L = Lx + Ly + Lz   
    gradL = gradLx + gradLy + dlnpdZ      
    gradL = np.reshape(gradL, (cygnet_likelihood.N * cygnet_likelihood.Q, ))   # reshape gradL back into 1D array   
    print(-2.*Lx, -2.*Ly, -2.*Lz)
    
    return -2.*L, -2.*gradL


def lnL_Z_old(pars, X, Y, hyper_params, Z_initial, X_var, Y_var, X_mask = None, Y_mask = None):  
    
    N = Z_initial.shape[0]
    Q = Z_initial.shape[1]
    D = X.shape[1]
    L = Y.shape[1]
    Z = np.reshape(pars, (N, Q)) 
    theta_rbf, theta_band, gamma_rbf, gamma_band = hyper_params    
    if X_mask is None:
        X_mask = np.ones_like(X).astype(bool)
    if Y_mask is None:
        Y_mask = np.ones_like(Y).astype(bool)
    
#    # fix variances to be the same at all pixels!
#    K1C = kernel1 + C_pix
#    factor1 = cho_factor(K1C, overwrite_a = True)
#    log_K1C_det = 2. * np.sum(np.log(np.diag(factor1[0])))
    
    Lx, Ly = 0., 0.
    gradLx = np.zeros_like(Z)
    gradLy = np.zeros_like(Z)
        
    for d in range(D):
        good_stars = X_mask[:, d]
        kernel1 = kernelRBF(Z, theta_rbf[d], theta_band)
        thiskernel = kernel1[good_stars, :][:, good_stars]
        K1C = thiskernel + np.diag(X_var[good_stars, d])
#        if nervous:
#            w = np.linalg.eigvalsh(K1C)
#            assert np.all(w > 0)
#            print w[0]/w[-1]
        thisfactor = cho_factor(K1C, overwrite_a = True)
        thislogdet = 2. * np.sum(np.log(np.diag(thisfactor[0])))
        Lx += LxOrLy(thislogdet, thisfactor, X[good_stars, d])   
        gradLx[good_stars, :] += dLdZ(X[good_stars, d], Z[good_stars, :], thisfactor, thiskernel, theta_band)        
    
    for l in range(L):
        good_stars = Y_mask[:, l]
        kernel2 = kernelRBF(Z, gamma_rbf[l], gamma_band)
        thiskernel = kernel2[good_stars, :][:, good_stars]
        K2C = thiskernel + np.diag(Y_var[good_stars, l])
        if nervous:
            w = np.linalg.eigvalsh(K2C)
            assert np.all(w > 0)
            print w[0]/w[-1]
        factor2 = cho_factor(K2C, overwrite_a = True)
        log_K2C_det = 2. * np.sum(np.log(np.diag(factor2[0])))
        Ly += LxOrLy(log_K2C_det, factor2, Y[good_stars, l]) 
        gradLy[good_stars, :] += dLdZ(Y[good_stars, l], Z[good_stars, :], factor2, thiskernel, gamma_band)    

    Lz = -0.5 * np.sum(Z**2)
    dlnpdZ = -Z 
    L = Lx + Ly + Lz   
    gradL = gradLx + gradLy + dlnpdZ      
    gradL = np.reshape(gradL, (N * Q, ))   # reshape gradL back into 1D array   
    print(-2.*Lx, -2.*Ly, -2.*Lz)
    
    return -2.*L, -2.*gradL


def LxOrLy(log_K_det, factor, data):  
    return -0.5 * log_K_det - 0.5 * np.dot(data, cho_solve(factor, data))


def UpdateFactor(factor, index, Z, theta_rbf, theta_band, K, X_var_d):
    
    # make sure we use upper matrix
    assert factor[1] == False

#    Z_new = np.array(Z)    
#    Z_new[index, :] = pars
    K_new = kernelRBF(Z, theta_rbf, theta_band)
    K_new[np.diag_indices_from(K_new)] += X_var_d
    u = K_new[index, :] - K[index, :]
    u[index] = 1.
    cholupdate(factor[0], u.copy())
    u[index] = 0.     
    choldowndate(factor[0], u)
    w = np.zeros_like(u)
    w[index] = 1.
    choldowndate(factor[0], w)
    
    return factor, K_new


def lnL_Z_one(pars, X, Y, X_var, Y_var, hyper_params, Z, all_factors, index_n): 

    theta_rbf, theta_band, gamma_rbf, gamma_band = hyper_params    
    K1_orig = kernelRBF(Z, theta_rbf, theta_band)
    K1 = np.array(K1_orig)
    
    Z[index_n, :] = pars
    #K2 = kernelRBF(Z, gamma_rbf, gamma_band) 
    
    D = X.shape[0]
    L = Y.shape[0]
    
    Lx, Ly = 0., 0.
    gradLx = np.zeros_like(Z)
    #gradLy = np.zeros_like(Z)
    for d in range(D):
        thisfactor, K1 = UpdateFactor(all_factors[d][0], index_n, Z, theta_rbf, theta_band, K1, X_var[:, d])
        thislogdet = 2. * np.sum(np.log(np.diag(thisfactor[0])))
        Lx += LxOrLy(thislogdet, thisfactor, X[:, d])
        gradLx += dLdZone(index_n, X, Z, thisfactor, K1_orig, theta_band)
        
#    for l in range(L):
#        thiskernel = K2[good_stars, :][:, good_stars]
#        K2C = thiskernel + np.diag(Y_var[good_stars, l])
#        #K2C = kernel2 + np.diag(Y_var[:, l])
#        factor2 = cho_factor(K2C, overwrite_a = True)
#        log_K2C_det = 2. * np.sum(np.log(np.diag(factor2[0])))
#        #K2C_inv = np.linalg.inv(K2C)
#        Ly += LxOrLy(log_K2C_det, factor2, Y[good_stars, l]) 
#        gradLy[good_stars, :] += dLdZ(Y[good_stars, l], Z[good_stars, :], factor2, thiskernel, gamma_band)    


    Lz = -0.5 * np.sum(pars**2)
    dlnpdZ = -pars
    L = Lx + Lz   
    gradL = gradLx + dlnpdZ      

    print(-2.*Lx, -2.*Ly, -2.*Lz)
    
    return -2.*L, -2.*gradL

# -------------------------------------------------------------------------------
# derivatives 
# -------------------------------------------------------------------------------

# derivative of the kernel with respect to the latent variables Z
def dKdZ(Z, K, band):
    grad_dKdZ = band * K[:, :, None, None] * A_matrix(Z)
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


def dLdK(K_inv, data):   
    grad_dLdK = -0.5 * K_inv + 0.5 * (np.dot(np.dot(K_inv, np.dot(data, data.T)), K_inv)) # first term * data.shape[1], if more than one dimension!
    return grad_dLdK                     # shape: N x N 
    

def dLdZ_old(data, Z, K_inv, K, band, good_stars = False):       
    grad_dKdZ = dKdZ(Z, K, band)
    grad_dLdK = dLdK(K_inv, data)
    #print grad_dKdZ.shape, grad_dLdK.shape
    #gradL = np.zeros_like(Z)
    #gradL[good_stars, :] = np.sum(grad_dLdK[:, :, None, None] * grad_dKdZ[good_stars, good_stars, good_stars, :], axis = (0, 1))
    #gradL = np.sum(grad_dLdK[:, :, None, None] * grad_dKdZ[good_stars, good_stars, :, :], axis = (0, 1))
    gradL = np.sum(grad_dLdK[:, :, None, None] * grad_dKdZ, axis = (0, 1))
    #print gradL.shape
    return gradL


def dLdZ_newer(data, Z, factor, K, band, good_stars = False): 
    #grad_dLdK = np.zeros_like(K)     
    #grad_dLdK[good_stars, :][:, good_stars] = dLdK(K_inv, data)
    grad_dLdK = dLdK(factor, data)
    gradL = np.zeros_like(Z)
    N, Q = Z.shape
    prefactor = grad_dLdK * band * K
    for l in range(N):
        foo = prefactor[:, l, None] * (Z[:, :] - Z[None, l, :])
        bar = prefactor[l, :, None] * (Z[:, :] - Z[l, None, :])
        gradL[l, :] += np.sum(foo + bar, axis = 0)        
    return gradL


def dLdZ(data, Z, factor, K, band): 
    
    K_inv_data = cho_solve(factor, data)
    prefactor = band * K

    gradL = np.zeros_like(Z)
    N, Q = Z.shape # N might not be the same as global N, if labels have been dropped
    for n in range(N):
        #print l, Z.shape, (Z[good_stars, :] - Z[l, :]).shape, prefactor.shape
        lhat = np.zeros((N))
        lhat[n] = 1.
        vec = prefactor[:, n, None] * (Z - Z[n, :])
        gradL[n, :] += K_inv_data[n] * np.dot(K_inv_data, vec) - np.dot(cho_solve(factor, lhat), vec)
    
    return gradL


def dLdZone(index, data, Z, factor, K, band):
    
    K_inv_data = cho_solve(factor, data)
    prefactor = band * K[:, index]
    N, Q = Z.shape 
    lhat = np.zeros((N))
    lhat[index] = 1.
    vec = prefactor[:, None] * (Z - Z[index, :])
    gradL = K_inv_data[index] * np.dot(K_inv_data, vec) - np.dot(cho_solve(factor, lhat), vec)
    
    return gradL

# -------------------------------------------------------------------------------
# testing derivatives
# -------------------------------------------------------------------------------

def test_dLdZone(index, data, Z_in, rbf, band, C_pix, eps):
    
    Z = np.array(Z_in)
    K = kernelRBF(Z, rbf, band) + C_pix
    factor = cho_factor(K)
    dL = dLdZone(index, data, Z, factor, K, band)
    
    for q in range(Z.shape[1]):
        Z[index, q] += eps
        K = kernelRBF(Z, rbf, band) + C_pix
        factor = cho_factor(K)
        log_K_det = 2. * np.sum(np.log(np.diag(factor[0])))
        Lplus = LxOrLy(log_K_det, factor, data)
        
        Z[index, q] -= 2. * eps
        K = kernelRBF(Z, rbf, band) + C_pix
        factor = cho_factor(K)
        log_K_det = 2. * np.sum(np.log(np.diag(factor[0])))
        Lminus = LxOrLy(log_K_det, factor, data)
        
        Z[index, q] += eps             
        xx = (Lplus - Lminus)/(2. * eps)             
        print dL[q], xx, (dL[q] - xx)/(dL[q] + xx)            
    return




def test_dLdK(K_in, data, eps):
    
    K = np.array(K_in)
    K_inv = np.linalg.inv(K)
    dL = dLdK(K_inv, data)
    
    for n in range(K.shape[0]):
        for m in range(K.shape[1]):
            K[n, m] += eps
            factor = cho_factor(K)
            log_K_det = 2. * np.sum(np.log(np.diag(factor[0])))
            Lplus = LxOrLy(log_K_det, factor, data)
            
            K[n, m] -= 2. * eps
            factor = cho_factor(K)
            log_K_det = 2. * np.sum(np.log(np.diag(factor[0])))
            Lminus = LxOrLy(log_K_det, factor, data)
            
            K[n, m] += eps
             
            print dL[n, m], (Lplus - Lminus)/(2. * eps)            
    return



def test_dLdZ(Z_in, data, rbf, band, C_pix, eps):
    
    Z = np.array(Z_in)
    K = kernelRBF(Z, rbf, band) + C_pix
    factor = cho_factor(K)
    dL = dLdZ(data, Z, factor, K, band, good_stars = False)
    
    for l in range(Z.shape[0]):
        for q in range(Z.shape[1]):
            Z[l, q] += eps
            K = kernelRBF(Z, rbf, band) + C_pix
            factor = cho_factor(K)
            log_K_det = 2. * np.sum(np.log(np.diag(factor[0])))
            Lplus = LxOrLy(log_K_det, factor, data)
            
            Z[l, q] -= 2. * eps
            K = kernelRBF(Z, rbf, band) + C_pix
            factor = cho_factor(K)
            log_K_det = 2. * np.sum(np.log(np.diag(factor[0])))
            Lminus = LxOrLy(log_K_det, factor, data)
            
            Z[l, q] += eps             
            xx = (Lplus - Lminus)/(2. * eps)             
            print dL[l, q], xx, (dL[l, q] - xx)/(dL[l, q] + xx)            
    return

# -------------------------------------------------------------------------------
# optimization of hyper parameters
# -------------------------------------------------------------------------------


def lnL_h(pars, X, Y, Z, Z_initial, X_var, Y_var, X_mask = None, Y_mask = None):
       
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
        
        if X_mask is None:
            X_mask = np.ones_like(X).astype(bool)
        if Y_mask is None:
            Y_mask = np.ones_like(Y).astype(bool)
        
        Lx, Ly = 0., 0.
        gradLx = np.zeros((2,))
        gradLy = np.zeros((2,))
        for d in range(D):
            good_stars = X_mask[:, d]
            thiskernel = kernel1[good_stars, :][:, good_stars]
            K1C = thiskernel + np.diag(X_var[good_stars, d])
            thisfactor = cho_factor(K1C, overwrite_a = True)
            thislogdet = 2. * np.sum(np.log(np.diag(thisfactor[0])))
            Lx += LxOrLy(thislogdet, thisfactor, X[good_stars, d])   
            gradLx += dLdhyper(X[good_stars, d], Z[good_stars, :], theta_band, theta_rbf, thiskernel, thisfactor) 
            
        for l in range(L):
            good_stars = Y_mask[:, l]
            thiskernel = kernel2[good_stars, :][:, good_stars]
            K2C = thiskernel + np.diag(Y_var[good_stars, l])
            thisfactor = cho_factor(K2C, overwrite_a = True)
            thislogdet = 2. * np.sum(np.log(np.diag(thisfactor[0])))
            Ly += LxOrLy(thislogdet, thisfactor, Y[good_stars, l]) 
            gradLy += dLdhyper(Y[good_stars, l], Z[good_stars, :], gamma_band, gamma_rbf, thiskernel, thisfactor) 
         
        #Lz = - 0.5 * np.sum(Z**2)                      
        L = Lx + Ly #+ Lz
        gradL = np.hstack((gradLx, gradLy))
        
        print(-2.*Lx, -2.*Ly) 
        return -2.*L, -2.*gradL
    else:
        print('hyper parameters negative!') 
        return np.inf, np.inf * np.ones_like(pars) # hack! check again!


def dLdhyper_old(data, Z, band, rbf, K, factor):  
    
    K_inv = np.linalg.inv(K)

    dKdrbf = 1./rbf * K  
    B = B_matrix(Z)
    dKdband = K * B
    
    dLdrbf = np.sum(dLdK(K_inv, data) * dKdrbf)        
    dLdband = np.sum(dLdK(K_inv, data) * dKdband)    
    
    return np.array([dLdrbf, dLdband])

    
def dLdhyper(data, Z, band, rbf, K, factor):  
    
    K_inv_data = cho_solve(factor, data) 
    
    dKdrbf = 1./rbf * K  
    B = B_matrix(Z)
    dKdband = K * B
    
    #K_inv_rbf = cho_solve(factor, dKdrbf)
    #dLdrbf = np.sum(-0.5 * K_inv_rbf + 0.5 * np.dot(np.dot(data, cho_solve(factor, data)) , K_inv_rbf))
    #dLdrbf = np.sum(-0.5 * K_inv_rbf + 0.5 * cho_solve(factor, np.dot(np.dot(data, data.T), K_inv_rbf)))
    
    #K_inv_band = cho_solve(factor, dKdband)
    #dLdband = np.sum(-0.5 * K_inv_band + 0.5 * cho_solve(factor, np.dot(np.dot(data, data.T), K_inv_band)))
    dLdband = np.sum(0.5 * K_inv_data * np.dot(K_inv_data, dKdband)) - 0.5 * np.trace(cho_solve(factor, dKdband))

    dLdrbf = np.sum(0.5 * K_inv_data * np.dot(K_inv_data, dKdrbf)) - 0.5 * np.trace(cho_solve(factor, dKdrbf))
    #dLdband = (-0.5 * np.trace(cho_solve(factor, dKdband)))
    #dLdrbf = (-0.5 * np.trace(cho_solve(factor, dKdrbf)))
    
    return np.array([dLdrbf, dLdband])
    
    
#def dLdhyper(data, Z, band, rbf, K, factor):     
#    
#    K_inv_data = cho_solve(factor, data)
#    B = B_matrix(Z)
#    dKdrbf = 1./rbf * K
#    dKdband = K * B
#    
#    gradL = np.zeros(Z)
#    N, Q = Z.shape # N might not be the same as global N, if labels have been dropped
#    for n in range(N):
#        lhat = np.zeros((N))
#        lhat[n] = 1.
#        vec = prefactor[:, n, None] * (Z - Z[n, :])
#        gradL[n, :] += K_inv_data[n] * np.dot(K_inv_data, vec) - np.dot(cho_solve(factor, lhat), vec)
#    
#    return gradL


# -------------------------------------------------------------------------------
# prediction
# -------------------------------------------------------------------------------

def mean_var(Z, Zj, data, data_var, rbf, band):
    N = Z.shape[0]
    B = np.zeros((N, ))
    for i in range(N):
        B[i] = -0.5 * np.dot((Z[i, :] - Zj).T, (Z[i, :] - Zj))   
    
    # prediction for test object: loop over d is in previous function
    if data.ndim == 1:
        K = kernelRBF(Z, rbf, band)
        KC = K + np.diag(data_var)
        k_Z_zj = rbf * np.exp(band * B)
        factor = cho_factor(KC, overwrite_a = True)
        mean = np.dot(data.T, cho_solve(factor, k_Z_zj))
        var = rbf - np.dot(k_Z_zj.T, cho_solve(factor, k_Z_zj))
        return mean, var, k_Z_zj, factor
    
    # prediction for training objects
    else:
        D = data.shape[1]
        mean_j = []
        var_j = []
        for d in range(D):
            K = kernelRBF(Z, rbf[d], band)
            KC = K + np.diag(data_var[:, d])
            k_Z_zj = rbf[d] * np.exp(band * B)
            factor = cho_factor(KC, overwrite_a = True)
            mean_j.append(np.dot(data[:, d].T, cho_solve(factor, k_Z_zj)))
            var_j.append(rbf[d] - np.dot(k_Z_zj.T, cho_solve(factor, k_Z_zj)))        
        return np.array(mean_j), np.array(var_j), k_Z_zj
    


def lnL_znew(pars, X_new_j, X_new_var_j, Z, data, data_var, rbf, band):    
    Zj = pars
    D = X_new_j.shape[0]
    
    L = 0.
    gradL = 0.
    for d in range(D):
        mean, var, k_Z_zj, factor = mean_var(Z, Zj, data[:, d], data_var[:, d], rbf[d], band)
        assert var > 0.
        
        L += -0.5 * np.dot((X_new_j[d] - mean).T, (X_new_j[d] - mean)) / \
                          (var + X_new_var_j[d]) - 0.5 * np.log(var + X_new_var_j[d]) 
        
        # do not use variance of model for the prediction for the moment...
        #L += -0.5 * np.dot((X_new_j[d] - mean).T, (X_new_j[d] - mean)) / \
        #                  (X_new_var_j[d]) - 0.5 * np.log( X_new_var_j[d])
        
        #dLdmu, dLdsigma2 = dLdmusigma2(X_new_j[d], mean, (X_new_var_j[d]))
        dLdmu, dLdsigma2 = dLdmusigma2(X_new_j[d], mean, (var + X_new_var_j[d]))
        dmudZ, dsigma2dZ = dmusigma2dZ(data[:, d], factor, Z, Zj, k_Z_zj, band)
        
        gradL += np.dot(dLdmu, dmudZ) + np.dot(dLdsigma2, dsigma2dZ)
    
    return -2.*L, -2.*gradL


def dLdmusigma2(X_new_j, mean, var):
    dLdmu = (X_new_j - mean) / var
    dLdsigma2 = -0.5 / var + 0.5 * np.dot((X_new_j - mean).T, (X_new_j - mean)) / var**2.
    return dLdmu, dLdsigma2


def dmusigma2dZ(data, factor, Z, Zj, k_Z_zj, band):
    term2 = k_Z_zj[:, None] * band * (Z - Zj)
    Kinv_term2 = cho_solve(factor, term2)
    dmudZ = np.dot(data.T, Kinv_term2)    
    
    dsigma2dZ = -2. * np.dot(k_Z_zj , Kinv_term2)    
    return dmudZ, dsigma2dZ


def predictX(X_new, X_new_var, X, X_var, Y, Y_var, Z_final, hyper_params, y0, z0):
    
    N, Q = Z_final.shape
    N, L = Y.shape
    theta_rbf, theta_band, gamma_rbf, gamma_band = hyper_params
    #K1 = kernelRBF(Z_final, theta_rbf, theta_band)
    #K2 = kernelRBF(Z_final, gamma_rbf, gamma_band)

    res = op.minimize(lnL_znew, x0 = z0, args = (X_new, X_new_var, Z_final, X, X_var, theta_rbf, theta_band), method = 'L-BFGS-B', jac = True, 
                   options={'gtol':1e-12, 'ftol':1e-12})   
    Z_new = res.x
    success_z = res.success
    print('latent variable optimization - success: {}'.format(res.success))
    
    res = op.minimize(lnL_ynew, x0 = y0, args = (Z_new, Z_final, Y, Y_var, gamma_rbf, gamma_band), method = 'L-BFGS-B', jac = True, 
                   options={'gtol':1e-12, 'ftol':1e-12})   
    Y_new = res.x
    success_y = res.success
    print('new labels optimization - success: {}'.format(res.success))        
    
    return Z_new, Y_new, success_z, success_y


def lnL_ynew(pars, Zj, Z, data, data_var, rbf, band):    
    
    lj = pars
    L = data.shape[1]
    
    like = 0.
    gradL = []
    for l in range(L):
        mean, var, k_Z_zj, factor = mean_var(Z, Zj, data[:, l], data_var[:, l], rbf[l], band)
        #assert var > 0.

        like += -0.5 * np.dot((lj[l] - mean).T, (lj[l] - mean)) #/ var - 0.5 * np.log(var) 
        
        gradL.append( -(lj[l] - mean)) #/var )
    
    return -2.*like, -2.*np.array(gradL)

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
        #tr_ivar_input[bad, x] = 1.
        tr_var_input[bad, x] = 1e8
    # remove one outlier in T_eff and [N/Fe]!
    bad = tr_label_input[:, 0] > 5200.
    tr_label_input[bad, 0] = np.median(tr_label_input[:, 0])
    #tr_ivar_input[bad, 0] = 1. 
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

# -------------------------------------------------------------------------------
# xxx
# -------------------------------------------------------------------------------






















