#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:58:25 2017

@author: eilers
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.optimize as op
import seaborn as sns
import time
import pickle
from astropy.table import Column

from functions_gplvm import lnL_Z, lnL_h, mean_var, kernelRBF, predictX, make_label_input, get_pivots_and_scales, PCAInitial

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
# load spectra and labels
# -------------------------------------------------------------------------------

# loading training labels
f = open('/Users/eilers/Dropbox/cygnet/data/training_labels_apogee_tgas.pickle', 'r')
training_labels = pickle.load(f)
f.close()

# loading normalized spectra
f = open('/Users/eilers/Dropbox/cygnet/data/apogee_spectra_norm.pickle', 'r')    
spectra = pickle.load(f)
f.close()

wl = spectra[:, 0, 0]
fluxes = spectra[:, :, 1].T
ivars = (1./(spectra[:, :, 2]**2)).T 
        

# remove duplicates       
foo, idx = np.unique(training_labels['APOGEE_ID'], return_index = True)
training_labels = training_labels[idx]
fluxes = fluxes[idx, :]
ivars = ivars[idx, :]
        
# data masking       
masking = training_labels['K'] < 0.
training_labels = training_labels[~masking]
fluxes = fluxes[~masking]
ivars = ivars[~masking]

# scaling of data and training labels?!

# -------------------------------------------------------------------------------
# # calculate K_MAG_ABS and Q
# -------------------------------------------------------------------------------

Q = 10**(0.2*training_labels['K']) * training_labels['parallax']/100.                    # assumes parallaxes is in mas
Q_err = training_labels['parallax_error'] * 10**(0.2*training_labels['K'])/100. 
Q = Column(Q, name = 'Q_MAG')
Q_err = Column(Q_err, name = 'Q_MAG_ERR')
training_labels.add_column(Q, index = 12)
training_labels.add_column(Q_err, index = 13)

# -------------------------------------------------------------------------------
# latex
# -------------------------------------------------------------------------------

latex = {}
latex["TEFF"] = r"$T_{\rm eff}$"
latex["LOGG"] = r"$\log g$"
latex["FE_H"] = r"$\rm [Fe/H]$"
latex["ALPHA_M"] = r"$[\alpha/\rm M]$"
latex["C_FE"] = r"$\rm [C/Fe]$"
latex["N_FE"] = r"$\rm [N/Fe]$"
latex["Q_MAG"] = r"$Q$"

labels = np.array(['TEFF', 'FE_H', 'LOGG']) #, 'ALPHA_M', 'Q_MAG', 'N_FE', 'C_FE'])
Nlabels = len(labels)
latex_labels = [latex[l] for l in labels]
tr_label_input, tr_var_input = make_label_input(labels, training_labels)
print(Nlabels, tr_label_input.shape, tr_var_input.shape, fluxes.shape, ivars.shape) 


pivots, scales = get_pivots_and_scales(tr_label_input)
tr_label_input_scaled = (tr_label_input - pivots[None, :]) / scales[None, :]
tr_var_input_scaled = tr_var_input / (scales[None, :]**2)
print('pivots: {}'.format(pivots))
print('scales: {}'.format(scales))

# -------------------------------------------------------------------------------
# constants
# -------------------------------------------------------------------------------

N = 50           # number of stars
D = 50          # number of pixels
Q = 3             # number latent dimensions, Q<=L
L = 3             # number of labels
pixel = np.arange(D)
wl_start = 1600   

theta_rbf, gamma_rbf = 1., 1. #np.pi, np.pi
theta_band, gamma_band = 1e-3, 0.1 #np.pi, np.pi 
#theta_rbf, gamma_rbf = 1.24584922e-01, 6.22569702e+01
#theta_band, gamma_band = 6.36293913e+02, 3.12092146e+02

name = '{0}_{1}_{2}_{3}_nohyper'.format(theta_rbf, theta_band, gamma_rbf, gamma_band)
date = '20-11-17'

# -------------------------------------------------------------------------------
# take random indices
# -------------------------------------------------------------------------------

np.random.seed(30)
ind_train = np.random.choice(np.arange(fluxes.shape[0]), (N,))

# -------------------------------------------------------------------------------
# input data
# -------------------------------------------------------------------------------

X = fluxes[ind_train, wl_start:wl_start+D] - 1.
X_var = 1./ivars[ind_train, wl_start:wl_start+D]
Y = tr_label_input_scaled[ind_train, :]
Y_var = tr_var_input_scaled[ind_train, :]

# not necessary?
#Y = np.zeros((N, Q))
#Y_var = np.ones((N, Q)) * 100.                      # large variance for missing labels...
#Y[:, L] = tr_label_input_scaled[ind_train, :]
#Y_var[:, L] = 1./tr_ivar_input_scaled[ind_train, :]


# -------------------------------------------------------------------------------
# initialize parameters
# -------------------------------------------------------------------------------

hyper_params = np.array([theta_rbf, theta_band, gamma_rbf, gamma_band])


Z_initial = np.zeros((N, Q)) 
Z_initial[:, :L] = Y[:]
#Z_initial = PCAInitial(X, Q)            # PCA initialize
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
#    res = op.minimize(lnL_h, x0 = hyper_params, args = (X, Y, Z, Z_initial, X_var, Y_var), method = 'L-BFGS-B', jac = True, 
#                      options={'gtol':1e-12, 'ftol':1e-12})
#    
#    # update hyperparameters
#    hyper_params = res.x
#    print('success: {}'.format(res.success))
#    print('new hyper parameters: {}'.format(res.x))
    
    # optimize Z
    print("optimizing latent parameters")
    res = op.minimize(lnL_Z, x0 = Z, args = (X, Y, hyper_params, Z_initial, X_var, Y_var), method = 'L-BFGS-B', jac = True, 
                      options={'gtol':1e-12, 'ftol':1e-12})
             
    # update Z
    Z = res.x
    print res
    print('success: {}'.format(res.success))
    # print('new Z: {}'.format(res.x))

t2 = time.time()
print('optimization in {} s.'.format(t2-t1))

Z_final = np.reshape(Z, (N, Q))
kernel1 = kernelRBF(Z_final, theta_rbf, theta_band)
kernel2 = kernelRBF(Z_final, gamma_rbf, theta_band)

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
plt.close()

# plot separately for all labels!
for i, l in enumerate(labels):
    q = 0
    plt.figure(figsize=(8, 6))
    plt.tick_params(axis=u'both', direction='in', which='both')
    cm = plt.cm.get_cmap('viridis')
    sc = plt.scatter(Z_final[:, q], Z_final[:, q+1], c = Y[:, i], marker = 'o', cmap = cm)
    cbar = plt.colorbar(sc)
    cbar.set_label(r'{}'.format(latex[l]), rotation=270, size=fsize, labelpad = 10)
    #plt.xlim(-4, 4)
    #plt.ylim(-4, 4)
    plt.xlabel(r'latent dimension {}'.format(q), fontsize = fsize)
    plt.ylabel(r'latent dimension {}'.format(q+1), fontsize = fsize)
    plt.title(r'#stars: {0}, #pixels: {1}, #labels: {2}, #latent dim.: {3}, $\theta_{{band}} = {4}$, $\theta_{{rbf}} = {5}$'.format(N, D, L, Q, theta_band, theta_rbf), fontsize=fsize)
    plt.savefig('plots/{0}/Latent2Label_{1}_{2}.png'.format(date, l, name))
    plt.close()

# self project
j = 2
mean, var, foo = mean_var(Z_final, Z_final[j, :], X, X_var, kernel1, theta_rbf, theta_band)
chi2 = np.sum((X[j, :] - mean)**2/X_var[j, :])
plt.figure(figsize=(8, 6))
plt.tick_params(axis=u'both', direction='in', which='both')
plt.plot(wl[wl_start:wl_start+D], X[j, :], label='original data', color='k')
plt.fill_between(wl[wl_start:wl_start+D], X[j, :] - 0.5*np.sqrt(X_var[j, :]), X[j, :] + 0.5*np.sqrt(X_var[j, :]), color='k', alpha = .3)
plt.plot(wl[wl_start:wl_start+D], mean, label=r'GPLVM model' + '\n $\chi^2\sim{}$'.format(round(chi2, 5)), color='r')
plt.fill_between(wl[wl_start:wl_start+D], mean - 0.5*np.sqrt(var), mean + 0.5*np.sqrt(var), color='r', alpha = .3)
plt.title(r'star {0}: $\theta_{{\rm band}} = {1},\,\theta_{{\rm rbf}} = {2}$'.format(ind_train[j], theta_band, theta_rbf), fontsize = fsize)
plt.legend(frameon = True)
plt.xlabel('wavelength', fontsize = fsize)
plt.ylabel('data', fontsize = fsize)
plt.savefig('plots/{0}/data_model_{1}.png'.format(date, name))
plt.close()       
       
# inferring labels!  
mean_labels = np.zeros_like(Y)
for j in range(N):   
    mean, var, foo = mean_var(Z_final, Z_final[j, :], Y, Y_var, kernel2, theta_rbf, theta_band)
    mean_labels[j, :] = mean

# rescale labels:
mean_labels_rescaled = np.zeros_like(mean_labels)
Y_rescaled = np.zeros_like(Y)
for l in range(L):
    mean_labels_rescaled[:, l] = mean_labels[:, l] * scales[l] + pivots[l]
    Y_rescaled[:, l] = Y[:, l] * scales[l] + pivots[l]
    

plot_limits = {}
plot_limits['TEFF'] = (3000, 7000)
plot_limits['FE_H'] = (-2.5, 1)
plot_limits['LOGG'] = (0, 2.1)
plot_limits['ALPHA_FE'] = (-.2, .6)
plot_limits['KMAG_ABS'] = (-1, -6)


for i, l in enumerate(labels):

    orig = Y_rescaled[:, i]
    gp_values = mean_labels_rescaled[:, i]
    scatter = np.round(np.std(orig - gp_values), 5)
    bias = np.round(np.mean(orig - gp_values), 5)    
    
    xx = [-10000, 10000]
    plt.figure(figsize=(6, 6))
    plt.scatter(orig, gp_values, color=colors[-2], label=' bias = {0} \n scatter = {1}'.format(bias, scatter), marker = 'o')
    plt.plot(xx, xx, color=colors[2], linestyle='--')
    plt.xlabel(r'reference labels {}'.format(latex[l]), size=fsize)
    plt.ylabel(r'inferred values {}'.format(latex[l]), size=fsize)
    plt.tick_params(axis=u'both', direction='in', which='both')
    plt.xlim(plot_limits[l])
    plt.ylim(plot_limits[l])
    plt.tight_layout()
    plt.legend(loc=2, fontsize=14, frameon=True)
    plt.title('#stars: {0}, #pixels: {1}, $\\theta_{{band}} = {4}$, $\\theta_{{rbf}} = {5}$'.format(N, D, L, Q, theta_band, theta_rbf), fontsize=fsize)
    plt.savefig('plots/{0}/1to1_{1}_{2}.png'.format(date, l, name))
    plt.close()


# -------------------------------------------------------------------------------
# prediction for new test object
# -------------------------------------------------------------------------------

print('prediction of test object:')

N_new = 1
# new testing object
ind_test = np.array([10, 50, 100, 188, 500, 1000, 1200, 1500, 1880, 2000]) #np.random.choice(np.arange(fluxes.shape[0]), (N_new,))
for i in range(len(ind_test)):
    ind = np.array([ind_test[i],])
    X_new = fluxes[ind, wl_start:wl_start+D] - 1.
    X_new_var = 1./ivars[ind, wl_start:wl_start+D]
    
    Z_new = predictX(N_new, X_new, X_new_var, X, X_var, Z_final, kernel1, hyper_params)
    #print Z_new
    
    j = 0
    mean_new, var_new, foo = mean_var(Z_final, Z_new[j, :], X, X_var, kernel1, theta_rbf, theta_band)
    chi2_test = np.sum((X_new[j, :] - mean_new)**2/X_new_var[j, :])
    plt.figure(figsize=(8, 6))
    plt.tick_params(axis=u'both', direction='in', which='both')
    plt.plot(wl[wl_start:wl_start+D], X_new[j, :], label='original data', color='k')
    plt.fill_between(wl[wl_start:wl_start+D], X_new[j, :] - 0.5*np.sqrt(X_new_var[j, :]), X_new[j, :] + 0.5*np.sqrt(X_new_var[j, :]), color='k', alpha = .3)
    plt.plot(wl[wl_start:wl_start+D], mean_new, label=r'GPLVM model' + '\n $\chi^2\sim{}$'.format(round(chi2_test, 4)), color='r')
    plt.fill_between(wl[wl_start:wl_start+D], mean_new - 0.5*np.sqrt(var_new), mean_new + 0.5*np.sqrt(var_new), color='r', alpha = .3)
    plt.title(r'star {0}: $\theta_{{\rm band}} = {1},\,\theta_{{\rm rbf}} = {2}$'.format(ind_test[0], theta_band, theta_rbf), fontsize = fsize)
    plt.legend(frameon = True)
    plt.xlabel('wavelength', fontsize = fsize)
    plt.ylabel('data', fontsize = fsize)
    plt.savefig('plots/{0}/testing_object_{1}_{2}.png'.format(date, name, ind[0]))
    plt.close()

## calculate distribution of chi^2 for all spectra... (or 100 randomly chosen ones...)
#np.random.seed(10)
#chis = []
#for i in range(100):
#    ind_test_i = np.random.choice(np.arange(len(fluxes)), (N_new,)) #np.array([ind_test[i]])
#    print ind_test_i
#    X_new = fluxes[ind_test_i, wl_start:wl_start+D]
#    X_new_var = 1./ivars[ind_test_i, wl_start:wl_start+D]    
#    Z_new = predictX(N_new, X_new, X_new_var, X, X_var, Z_final, kernel1, hyper_params)    
#    j = 0
#    mean_new, var_new, foo = mean_var(Z_final, Z_new[j, :], X, X_var, kernel1, theta_rbf, theta_band)
#    chi2_test = np.sum((X_new[j, :] - mean_new)**2/X_new_var[j, :])
#    chis.append(chi2_test)
#plt.hist(chis, label = r'$\langle \chi^2\rangle = {0}$'.format(np.mean(chis)))
#plt.tick_params(axis=u'both', direction='in', which='both')
#plt.xlabel(r'$\chi^2$')
#plt.ylabel('counts')
#plt.legend(frameon=True)
#plt.title(r'$\theta_{{rbf}} = {0}, \theta_{{band}} = {1}, \gamma_{{rbf}} = {2}, \gamma_{{band}} = {3}$'.format(theta_rbf, theta_band, gamma_rbf, gamma_band))
#plt.savefig('plots/{0}/chis_{1}.png'.format(date, name))
   

# -------------------------------------------------------------------------------
# testing derivatives...
# -------------------------------------------------------------------------------'''

'''zz = np.array([[-1.80532781, -2.62380071,  0.47815664],
       [ 1.71816682, -0.3423434 , -2.8154128 ],
       [ 2.52546325, -0.33267094,  2.5913645 ],
       [-1.43773614,  2.78122921,  0.59679351],
       [-3.49598149,  0.38773783, -3.73423   ]])
    
#zz = np.reshape(zz, (15,))

zz2 = np.array([[-1.80532781, -2.62380071,  0.47815664],
       [ 1.71816682, -0.3423434 , -2.8154128 ],
       [ 2.52546325, -0.33267094,  2.5913645 ],
       [-1.43773614,  2.78122921,  0.59679351],
       [-3.49598149,  0.38773783, -3.73423   ]])

#zz2 = np.reshape(zz2[0, :], (3,))
tiny = 1e-3
q = 0
zz2[0, 0] += tiny
   
   
L1 = -0.5 * np.sum(zz**2)
L2 = -0.5 * np.sum(zz2**2)
gradL = -zz
print ((L2 - L1)/tiny - gradL[0])/((L2 - L1)/tiny + gradL[0])
   
L1, gradL1, Lx, dLx, Ly, dLy, Lz, dLz = lnL_Z(np.reshape(zz, (15,)), X, Y, hyper_params, Z_initial, X_var, Y_var)
L2, gradL2, Lx2, dLx2, Ly2, dLy2, Lz2, dLz2 = lnL_Z(np.reshape(zz2, (15,)), X, Y, hyper_params, Z_initial, X_var, Y_var)

print ((L2 - L1)/tiny - gradL1[q]) / ((L2 - L1)/tiny + gradL1[q])
print ((Lx2 - Lx)/tiny - dLx[q]) / ((Lx2 - Lx)/tiny + dLx[q])
print ((Ly2 - Ly)/tiny - dLy[q]) / ((Ly2 - Ly)/tiny + dLy[q])
print ((Lz2 - Lz)/tiny - dLz[q]) / ((Lz2 - Lz)/tiny + dLz[q])
# -------------------------------------------------------------------------------


kernel1 = kernelRBF(Z_initial, theta_rbf, theta_band)
kernel2 = kernelRBF(Z_initial, gamma_rbf, gamma_band)

#K1C = kernel1 + var_d * np.eye(N)
#K2C = kernel2 + var_d * np.eye(N)

Lx, Ly, gradLx, gradLy = 0., 0., 0., 0.
for d in range(D):
    K1C = kernel1 + np.diag(X_var[:, d])
    Lx += LxOrLy(K1C, X[:, d])   
    gradLx += dLdhyper(X[:, d], Z_initial, theta_band, theta_rbf, kernel1, K1C) 
    
for l in range(L):
    K2C = kernel2 + np.diag(Y_var[:, l])
    Ly += LxOrLy(K2C, Y[:, l]) 
    gradLy += dLdhyper(Y[:, l], Z_initial, gamma_band, gamma_rbf, kernel1, K2C)

gamma_rbf += 1e-5

kernel1 = kernelRBF(Z_initial, theta_rbf, theta_band)
kernel2 = kernelRBF(Z_initial, gamma_rbf, gamma_band)

#K1C = kernel1 + var_d * np.eye(N)
#K2C = kernel2 + var_d * np.eye(N)

Lx2, Ly2, gradLx, gradLy = 0., 0., 0., 0.
for d in range(D):
    K1C = kernel1 + np.diag(X_var[:, d])
    Lx2 += LxOrLy(K1C, X[:, d])   
    gradLx += dLdhyper(X[:, d], Z_initial, theta_band, theta_rbf, kernel1, K1C) 
    
for l in range(L):
    K2C = kernel2 + np.diag(Y_var[:, l])
    Ly2 += LxOrLy(K2C, Y[:, l]) 
    gradLy += dLdhyper(Y[:, l], Z_initial, gamma_band, gamma_rbf, kernel1, K2C)


# -------------------------------------------------------------------------------'''

