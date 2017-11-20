#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 14:31:26 2017

@author: eilers
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.optimize as op
import seaborn as sns
import time

from functions_gplvm import lnL_Z, lnL_h, mean_var, kernelRBF, predictX

# -------------------------------------------------------------------------------
# plotting settings
# -------------------------------------------------------------------------------

colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple", "pale red", "orange", "red", "blue"]
colors = sns.xkcd_palette(colors)
sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white', 'axes.edgecolor':'black', 'xtick.direction': 'in', 'ytick.direction': 'in'})
sns.set_style("ticks")
matplotlib.rcParams['ytick.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 14
fsize = 14

# -------------------------------------------------------------------------------
# constants
# -------------------------------------------------------------------------------

N = 5             # number of stars
D = 7             # number of pixels
Q = 3             # number latent dimensions, Q<=L
L = 3             # number of labels
pixel = np.arange(D)

# -------------------------------------------------------------------------------
# data
# -------------------------------------------------------------------------------

np.random.seed(42)
X = np.random.uniform(0, 1, (N, D))          # data
Y = np.random.normal(0, 1, (N, L))           # labels

# uncertainties! 
var_d = 0.01
X_var = var_d * np.ones_like(X)
#X_var[1, :] = 0.1
Y_var = 0.01 * np.ones_like(Y)

# -------------------------------------------------------------------------------
# initialize parameters
# -------------------------------------------------------------------------------

theta_rbf, gamma_rbf = 1.3, 1.3 #np.pi, np.pi
hyper_params = np.array([theta_rbf, gamma_rbf])

# PCA initialize?
Z_initial = Y[:] 
Z = np.reshape(Z_initial, (N*Q,))

# -------------------------------------------------------------------------------
# optimization
# -------------------------------------------------------------------------------

print('initial hyper parameters: %s' %hyper_params)
print('initial latent variables: %s' %Z)

t1 = time.time()

max_iter = 1
for t in range(max_iter):
    
#    # optimize hyperparameters
#    print("optimizing hyper parameters")
#    res = op.minimize(lnL_h, x0 = hyper_params, args = (X, Y, Z, Z_initial), method = 'L-BFGS-B', jac = True, 
#                      options={'gtol':1e-12, 'ftol':1e-12})
    
    # update hyperparameters
#    hyper_params = res.x
#    print('success: {}'.format(res.success))
#    print('new hyper parameters: {}'.format(res.x))
             
    # optimize Z
    print("optimizing latent parameters")
    res = op.minimize(lnL_Z, x0 = Z, args = (X, Y, hyper_params, Z_initial, X_var, Y_var), method = 'L-BFGS-B', jac = True, 
                      options={'gtol':1e-12, 'ftol':1e-12})
             
    # update Z
    Z = res.x
    print('success: {}'.format(res.success))
    # print('new Z: {}'.format(res.x))

t2 = time.time()
print('optimization in {} s.'.format(t2-t1))

Z_final = np.reshape(Z, (N, Q))
kernel1 = kernelRBF(Z_final, theta_rbf)
kernel2 = kernelRBF(Z_final, gamma_rbf)

# -------------------------------------------------------------------------------
# visualisation
# -------------------------------------------------------------------------------

# plot separately for all labels!
plt.figure(figsize=(6, 6))
plt.tick_params(axis=u'both', direction='in', which='both')
plt.scatter(np.reshape(Z_initial, (N*Q,)), Z, zorder=10)
plt.plot([-5, 5], [-5, 5], linestyle = '--', color=colors[2], zorder=5)
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.xlabel('initial labels', fontsize = fsize)
plt.ylabel('inferred labels', fontsize = fsize)
plt.title('iteration {}'.format(t+1), fontsize = fsize)

# self project
j = 1
mean, var, foo = mean_var(Z_final, Z_final[j, :], X, X_var, kernel1, theta_rbf)
plt.figure(figsize=(8, 6))
plt.tick_params(axis=u'both', direction='in', which='both')
plt.plot(X[j, :], label='original data', color='k')
plt.fill_between(pixel, X[j, :] - 0.5*np.sqrt(X_var[j, :]), X[j, :] + 0.5*np.sqrt(X_var[j, :]), color='k', alpha = .3)
plt.plot(mean, label=r'GPLVM model' + '\n $\Delta\sim{}$'.format(np.mean(X[j, :] - mean)), color='r')
plt.fill_between(pixel, mean - 0.5*np.sqrt(var), mean + 0.5*np.sqrt(var), color='r', alpha = .3)
plt.title(r'star {0}: $\theta_{{\rm band}} = {1},\,\theta_{{\rm rbf}} = {2},\,\sigma_x^2 = {3}$'.format(j+1, 1, theta_rbf, var_d), fontsize = fsize)
plt.legend(frameon = True)
plt.xlabel('pixel', fontsize = fsize)
plt.ylabel('data', fontsize = fsize)
#plt.savefig('plots/data_model_11.pdf')
#print X[j, :] - mean
       
       
# inferring labels!  
mean_labels = np.zeros_like(Y)
for j in range(N):   
    mean, var, foo = mean_var(Z_final, Z_final[j, :], Y, Y_var, kernel2, theta_rbf)
    mean_labels[j, :] = mean

l = 0
bias = round(np.mean(Y[:, l] - mean_labels[:, l]), 4)
scatter =round(np.std(Y[:, l] - mean_labels[:, l]), 4)
plt.figure(figsize=(6, 6))
plt.tick_params(axis=u'both', direction='in', which='both')
plt.scatter(Y[:, l], mean_labels[:, l], label=' bias = {0} \n scatter = {1}'.format(bias, scatter), s=100)
plt.plot([-10, 10], [-10, 10], linestyle = '--', color = colors[2])
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.legend(frameon = True, fontsize = fsize)
plt.title(r'label #{}'.format(l+1), fontsize = fsize)
plt.xlabel('original labels', fontsize = fsize)
plt.ylabel('infered labels', fontsize = fsize)


# -------------------------------------------------------------------------------
# prediction for new test object
# -------------------------------------------------------------------------------
            
print('prediction of test object:')

# new testing object
np.random.seed(5)
N_new = 1
X_new = np.random.uniform(0, 1, (N_new, D))
X_new_var = 0.01 * np.ones_like(X_new)

Z_new = predictX(N_new, X_new, X_new_var, X, X_var, Z_final, kernel1, hyper_params)

j = 0
mean_new, var_new, foo = mean_var(Z_final, Z_new[j, :], X, X_var, kernel1, theta_rbf)
chi2_test = np.sum((X_new[j, :] - mean_new)**2/X_new_var[j, :])
plt.figure(figsize=(8, 6))
plt.tick_params(axis=u'both', direction='in', which='both')
plt.plot(X_new[j, :], color = 'k', label='data: new object')
plt.fill_between(pixel, X_new[j, :] - np.sqrt(X_new_var[j, :]), X_new[j, :] + np.sqrt(X_new_var[j, :]), color='k', alpha = .3)
plt.plot(mean_new, color = 'r', label='GPLVM model')
plt.fill_between(pixel, mean_new - np.sqrt(var_new), mean_new + np.sqrt(var_new), color='r', alpha = .3)
plt.legend(frameon = True)
plt.xlabel('wavelength', fontsize = fsize)
plt.ylabel('data', fontsize = fsize)
#plt.savefig('plots/{0}/testing_object_{1}.pdf'.format(date, name))

# -------------------------------------------------------------------------------
# xxx
# -------------------------------------------------------------------------------

# testing
'''tiny = 1e-5
Znew = np.reshape(Z, (N, Q)) 
kernel1 = kernelRBF(Znew, theta_rbf)
Lx, Ly, gradLx, gradLy = 0., 0., 0., 0.
for d in range(D):
    K1C = kernel1 + np.diag(X_var[:, d])
    Lx += LxOrLy(K1C, X[:, d])   
    gradLx += dLdZ(X[:, d], Znew, K1C, theta_rbf)
gradLx = np.reshape(gradLx, (N * Q, ))   # reshape gradL back into 1D array   


Z2 = 1. * Z
Z2[1] += tiny
print Z2 - Z
Znew2 = np.reshape(Z2, (N, Q)) 
kernel1 = kernelRBF(Znew2, theta_rbf)
Lx2, Ly2, gradLx2, gradLy2 = 0., 0., 0., 0.
for d in range(D):
    K2C = kernel1 + np.diag(X_var[:, d])
    Lx2 += LxOrLy(K2C, X[:, d])   
    gradLx2 += dLdZ(X[:, d], Znew2, K2C, theta_rbf)
    
print Lx, Lx2, (Lx2-Lx)/tiny, gradLx[1], ((Lx2-Lx)/tiny- gradLx[1])/ ((Lx2-Lx)/tiny+ gradLx[1])


kernel2 = kernelRBF(Znew, theta_rbf)
for l in range(L):
    K2C = kernel2 + np.diag(Y_var[:, l])
    Ly += LxOrLy(K2C, Y[:, l]) 
    gradLy += dLdZ(Y[:, l], Znew, K2C, gamma_rbf)
gradLy = np.reshape(gradLy, (N * Q, ))   # reshape gradL back into 1D array   
    
kernel2 = kernelRBF(Znew2, theta_rbf)
for l in range(L):
    K2C = kernel2 + np.diag(Y_var[:, l])
    Ly2 += LxOrLy(K2C, Y[:, l]) 
    gradLy2 += dLdZ(Y[:, l], Znew2, K2C, gamma_rbf)

print Ly, Ly2, (Ly2-Ly)/tiny, gradLy[1], ((Ly2-Ly)/tiny- gradLy[1])/ ((Ly2-Ly)/tiny+ gradLy[1])


l, dl, lx, dx, ly, dy, lz, dz = lnL_Z(Z, X, Y, hyper_params, Z_initial, X_var, Y_var)
l2, dl2, lx2, dx2, ly2, dy2, lz2, dz2 = lnL_Z(Z2, X, Y, hyper_params, Z_initial, X_var, Y_var)

print l, l2, (l2-l)/tiny, dl[1]
print lx, lx2, (lx2-lx)/tiny, dx[1], ((lx2-lx)/tiny - dx[1])/((lx2-lx)/tiny + dx[1])
print ly, ly2, (ly2-ly)/tiny, dy[1], ((ly2-ly)/tiny - dy[1])/((ly2-ly)/tiny + dy[1])
print lz, lz2, (lz2-lz)/tiny, dz[1], ((lz2-lz)/tiny - dz[1])/((lz2-lz)/tiny + dz[1])

# -------------------------------------------------------------------------------'''


