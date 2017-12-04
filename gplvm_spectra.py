#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:58:25 2017

@author: eilers
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import scipy.optimize as op
import seaborn as sns
import time
import pickle
from astropy.table import Column, Table, join

from functions_gplvm import lnL_Z, lnL_h, mean_var, kernelRBF, predictX, make_label_input, get_pivots_and_scales, PCAInitial
from NN import Chi2_Matrix, NN

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
# load spectra and labels (Red Clump Stars)
# -------------------------------------------------------------------------------

## loading training labels
##f = open('/Users/eilers/Dropbox/cygnet/data/training_labels_apogee_tgas.pickle', 'r')
#f = open('data/training_labels_apogee_tgas.pickle', 'r')
#training_labels = pickle.load(f)
#f.close()
#
## loading normalized spectra
##f = open('/Users/eilers/Dropbox/cygnet/data/apogee_spectra_norm.pickle', 'r')
#f = open('data/apogee_spectra_norm.pickle', 'r')        
#spectra = pickle.load(f)
#f.close()
#
#wl = spectra[:, 0, 0]
#fluxes = spectra[:, :, 1].T
#ivars = (1./(spectra[:, :, 2]**2)).T 
#        
#
## remove duplicates       
#foo, idx = np.unique(training_labels['APOGEE_ID'], return_index = True)
#training_labels = training_labels[idx]
#fluxes = fluxes[idx, :]
#ivars = ivars[idx, :]
#        
## data masking       
#masking = training_labels['K'] < 0.
#training_labels = training_labels[~masking]
#fluxes = fluxes[~masking]
#ivars = ivars[~masking]

# -------------------------------------------------------------------------------
# load spectra and labels (cluster stars)
# -------------------------------------------------------------------------------

table = Table.read('data/meszaros13_table4_mod_new.txt', format='ascii', data_start = 0) 
table['col3'].name = 'CLUSTER'

f = open('data/train_labels_feh.pickle', 'r')    
training_labels = pickle.load(f)
f.close()
training_labels = Table(training_labels)
training_labels['col0'].name = 'TEFF'
training_labels['col1'].name = 'TEFF_ERR'
training_labels['col2'].name = 'LOGG'
training_labels['col3'].name = 'LOGG_ERR'
training_labels['col4'].name = 'FE_H'
training_labels['col5'].name = 'FE_H_ERR'

# mask all spectra with a -9999.9 entry! (all spectra have [alpha/Fe] measured)
missing = np.logical_or(training_labels['TEFF'] < -9000., training_labels['LOGG'] < -9000., training_labels['FE_H'] < -9000.)
training_labels = training_labels[~missing]
table = table[~missing]

f = open('data/all_data_norm.pickle', 'r')    
spectra = pickle.load(f)
f.close()
spectra = spectra[:, ~missing, :]
wl = spectra[:, 0, 0]
fluxes = spectra[:, :, 1].T
ivars = (1./(spectra[:, :, 2]**2)).T  
        
# exclude pleiades
pleiades = table['CLUSTER'] == 'Pleiades'
training_labels = training_labels[~pleiades]
fluxes = fluxes[~pleiades]
ivars = ivars[~pleiades]

## -------------------------------------------------------------------------------
## load spectra and labels (cluster stars)
## -------------------------------------------------------------------------------
#
#create_data_set = True
#
#table = Table.read('data/meszaros13_table4.txt', format='ascii', data_start = 0) 
#table['col2'].name = 'CLUSTER'
#table['col1'].name = 'APOGEE_ID'
#
#     
## exclude pleiades
#pleiades = table['CLUSTER'] == 'Pleiades'
#table = table[~pleiades]
#
#if create_data_set:
#    
#    # open APOGEE data
#    apogee = Table.read('./data/allStar-l31c.2.fits', format='fits', hdu = 1)
#    joined_data = join(table, apogee, keys = 'APOGEE_ID', join_type='left')
#
#    foo, idx = np.unique(joined_data['APOGEE_ID'], return_index = True)
#    training_labels = joined_data[idx]
#    f = open('data/train_labels_all.pickle', 'w')  
#    pickle.dump(training_labels, f)
#    f.close()
#
#else:
#    f = open('data/train_labels_all.pickle', 'r')    
#    training_labels = pickle.load(f)
#    f.close()
#    
## set all missing labels to IVAR = 0.
#
## mask all spectra with a -9999.9 entry! (all spectra have [alpha/Fe] measured)
#missing = np.logical_or(training_labels['TEFF'] < -9000., training_labels['LOGG'] < -9000., training_labels['FE_H'] < -9000.)
#training_labels = training_labels[~missing]
#table = table[~missing]
#
#f = open('data/all_data_norm.pickle', 'r')    
#spectra = pickle.load(f)
#f.close()
#spectra = spectra[:, ~missing, :]
#wl = spectra[:, 0, 0]
#fluxes = spectra[:, :, 1].T
#ivars = (1./(spectra[:, :, 2]**2)).T  


# -------------------------------------------------------------------------------
# # calculate K_MAG_ABS and Q
# -------------------------------------------------------------------------------

#Q = 10**(0.2*training_labels['K']) * training_labels['parallax']/100.                    # assumes parallaxes is in mas
#Q_err = training_labels['parallax_error'] * 10**(0.2*training_labels['K'])/100. 
#Q = Column(Q, name = 'Q_MAG')
#Q_err = Column(Q_err, name = 'Q_MAG_ERR')
#training_labels.add_column(Q, index = 12)
#training_labels.add_column(Q_err, index = 13)

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

plot_limits = {}
plot_limits['TEFF'] = (3000, 7000)
plot_limits['FE_H'] = (-2.5, 1)
plot_limits['LOGG'] = (0, 4.)
plot_limits['ALPHA_FE'] = (-.2, .6)
plot_limits['KMAG_ABS'] = (-1, -6)
plot_limits['Q_MAG'] = (-3, 1)

labels = np.array(['TEFF', 'LOGG', 'FE_H']) #, 'ALPHA_M', 'Q_MAG', 'N_FE', 'C_FE'])
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

N = 50            # number of stars
D = 100           # number of pixels
Q = 6             # number latent dimensions, Q<=L
L = len(labels)             # number of labels
pixel = np.arange(D)
wl_start = 500   

theta_rbf, gamma_rbf = 0.1, 1. 
theta_band, gamma_band = 1e-4, 1e-4 

name = '{0}_{1}_{2}_{3}_Q{4}_D{5}_N{6}'.format(theta_rbf, theta_band, gamma_rbf, gamma_band, Q, D, N)
date = 'speed' # 'Q_tests' #

# -------------------------------------------------------------------------------
# take random indices
# -------------------------------------------------------------------------------

np.random.seed(30)
ind_train = np.random.choice(np.arange(fluxes.shape[0]), (N,))

# make a rand_label = np.random(size = N)

# -------------------------------------------------------------------------------
# input data
# -------------------------------------------------------------------------------

# subtract median spectrum from fluxes
X_orig = fluxes[ind_train, wl_start:wl_start+D] 
X_mean = np.median(X_orig, axis = 0)        
X = X_orig - X_mean

# plot the mean!          
          
X_var = 1./ivars[ind_train, wl_start:wl_start+D]
Y = tr_label_input_scaled[ind_train, :]
Y_var = tr_var_input_scaled[ind_train, :]

# -------------------------------------------------------------------------------
# construct one covariance matrix for all pixels with median inverse covariance for each star
# -------------------------------------------------------------------------------

med_var = np.median(X_var, axis = 1)
C_pix = np.diag(med_var)

# inverse!
#med_inv_var = np.median(1./X_var, axis = 1)
#C_inv = np.diag(med_inv_var)

# -------------------------------------------------------------------------------
# contrsuct NxD boolean object which is True, if data is good. 
# -------------------------------------------------------------------------------

X_mask = np.ones((N, D), dtype = bool)
for n in range(N):
    for d in range(D):
        if 1./X_var[n, d] < 10.:
            X_mask[n, d] = False

Y_mask = np.ones((N, L), dtype = bool)
for n in range(N):
    for l in range(L):
        if 1./Y_var[n, l] < 1.:
            Y_mask[n, l] = False
                       
#good_labels[:, 0] = False
#good_labels[10, 1] = False

plt.hist(np.sum(X_mask, axis = 0), bins = 50)
#plt.yscale('log')
plt.xlabel('# of missing pixels')
plt.tick_params(axis=u'both', direction='in', which='both')
plt.savefig('plots/{0}/bad_pixels_{1}.png'.format(date, name))
plt.close()

plt.hist(np.sum(Y_mask, axis = 1))
plt.xlabel('# of missing labels')
plt.xlim(0, L)
#plt.yscale('log')
plt.tick_params(axis=u'both', direction='in', which='both')
plt.savefig('plots/{0}/bad_labels_{1}.png'.format(date, name))
plt.close()


# -------------------------------------------------------------------------------
# initialize parameters
# -------------------------------------------------------------------------------

hyper_params = np.array([theta_rbf, theta_band, gamma_rbf, gamma_band])


#Z_initial = np.zeros((N, Q)) 
#Z_initial[:, :L] = Y[:]
Z_initial = PCAInitial(X, Q)            # PCA initialize, use mean as first component, i.e. 1 as initial guess for Z_initial for each star
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
#    res = op.minimize(lnL_h, x0 = hyper_params, args = (X, Y, Z, Z_initial, X_var, Y_var, C_pix), method = 'L-BFGS-B', jac = True, 
#                      options={'gtol':1e-12, 'ftol':1e-12})
#    
#    # update hyperparameters
#    hyper_params = res.x
#    print(res)
#    print('success: {}'.format(res.success))
#    print('new hyper parameters: {}'.format(res.x))
    
    # optimize Z
    print("optimizing latent parameters")
    res = op.minimize(lnL_Z, x0 = Z, args = (X, Y, hyper_params, Z_initial, X_var, Y_var, C_pix, X_mask, Y_mask), method = 'L-BFGS-B', jac = True, 
                      options={'gtol':1e-12, 'ftol':1e-12})
             
    # update Z
    Z = res.x
    print(res)
    print('success: {}'.format(res.success))
    # print('new Z: {}'.format(res.x))

t2 = time.time()
print('optimization in {} s.'.format(t2-t1))

Z_final = np.reshape(Z, (N, Q))
kernel1 = kernelRBF(Z_final, theta_rbf, theta_band)
kernel2 = kernelRBF(Z_final, gamma_rbf, theta_band)

# -------------------------------------------------------------------------------
# inferring new labels of training objects  
# -------------------------------------------------------------------------------

mean_labels = np.zeros_like(Y)
for j in range(N):   
    mean, var, foo = mean_var(Z_final, Z_final[j, :], Y, Y_var, kernel2, gamma_rbf, gamma_band)
    mean_labels[j, :] = mean

# rescale labels:
mean_labels_rescaled = np.zeros_like(mean_labels)
Y_rescaled = np.zeros_like(Y)
for l in range(L):
    mean_labels_rescaled[:, l] = mean_labels[:, l] * scales[l] + pivots[l]
    Y_rescaled[:, l] = Y[:, l] * scales[l] + pivots[l]
    
# -------------------------------------------------------------------------------
# visualisation
# -------------------------------------------------------------------------------

# latent space color coded by labels
for i, l in enumerate(labels):
    q = 0
    plt.figure(figsize=(9, 6))
    plt.tick_params(axis=u'both', direction='in', which='both')
    cm = plt.cm.get_cmap('viridis')
    sc = plt.scatter(Z_final[:, q], Z_final[:, q+1], c = Y[:, i], marker = 'o', cmap = cm)
    cbar = plt.colorbar(sc)
    cbar.set_label(r'{}'.format(latex[l]), rotation=270, size=fsize, labelpad = 10)
    #plt.xlim(-4, 4)
    #plt.ylim(-4, 4)
    plt.xlabel(r'latent dimension {}'.format(q), fontsize = fsize)
    plt.ylabel(r'latent dimension {}'.format(q+1), fontsize = fsize)
    plt.title(r'#stars: {0}, #pixels: {1}, #labels: {2}, #latent dim.: {3}, $\theta_{{band}} = {4}$, $\theta_{{rbf}} = {5}$, $\gamma_{{band}} = {6}$, $\gamma_{{rbf}} = {7}$'.format(N, D, L, Q, theta_band, theta_rbf, gamma_band, gamma_rbf), fontsize=12)
    plt.tight_layout()
    plt.savefig('plots/{0}/Latent2Label_{1}_{2}.png'.format(date, l, name))
    plt.close()

# data and model for training object
j = 2
mean, var, foo = mean_var(Z_final, Z_final[j, :], X, X_var, kernel1, theta_rbf, theta_band)
chi2 = np.sum((X[j, :] - mean)**2/X_var[j, :])
#plt.figure(figsize=(8, 6))
fig, ax = plt.subplots(ncols = 2, nrows = 1, figsize = (16, 6))
plt.title(r'star {0}: $\theta_{{\rm band}} = {1},\,\theta_{{\rm rbf}} = {2}$'.format(ind_train[j], theta_band, theta_rbf), fontsize = fsize)
ax[0].tick_params(axis=u'both', direction='in', which='both')
ax[0].plot(wl[wl_start:wl_start+D], X_orig[j, :], label='original data', color='k')
ax[0].fill_between(wl[wl_start:wl_start+D], X_orig[j, :] - 0.5*np.sqrt(X_var[j, :]), X_orig[j, :] + 0.5*np.sqrt(X_var[j, :]), color='k', alpha = .3)
ax[0].plot(wl[wl_start:wl_start+D], mean + X_mean, label=r'GPLVM model' + '\n $\chi^2\sim{}$'.format(round(chi2, 5)), color='r')
ax[0].fill_between(wl[wl_start:wl_start+D], mean + X_mean - 0.5*np.sqrt(var), mean + X_mean + 0.5*np.sqrt(var), color='r', alpha = .3)
ax[0].legend(frameon = True)
ax[0].set_xlabel('wavelength', fontsize = fsize)
ax[0].set_ylabel('data', fontsize = fsize)
ax[0].set_ylim(0, 1.2)
ax[1].set_ylim(-.2, .2)
ax[1].tick_params(axis=u'both', direction='in', which='both')
ax[1].set_xlabel('wavelength', fontsize = fsize)
ax[1].plot(wl[wl_start:wl_start+D], X[j, :], label='original data', color='k')
ax[1].fill_between(wl[wl_start:wl_start+D], X[j, :] - 0.5*np.sqrt(X_var[j, :]), X[j, :] + 0.5*np.sqrt(X_var[j, :]), color='k', alpha = .3)
ax[1].plot(wl[wl_start:wl_start+D], mean, label=r'GPLVM model' + '\n $\chi^2\sim{}$'.format(round(chi2, 5)), color='r')
ax[1].fill_between(wl[wl_start:wl_start+D], mean - 0.5*np.sqrt(var), mean + 0.5*np.sqrt(var), color='r', alpha = .3)
plt.savefig('plots/{0}/data_model_{1}_{2}.png'.format(date, name, j))
plt.close()       
    

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
    plt.title('#stars: {0}, #pixels: {1}, $\\theta_{{band}} = {4}$, $\\theta_{{rbf}} = {5}$, $\gamma_{{band}} = {6}$, $\gamma_{{rbf}} = {7}$'.format(N, D, L, Q, theta_band, theta_rbf, gamma_band, gamma_rbf), fontsize=12)
    plt.savefig('plots/{0}/1to1_{1}_{2}.png'.format(date, l, name))
    plt.close()

# -------------------------------------------------------------------------------
# prediction for new test object
# -------------------------------------------------------------------------------

print('prediction of test object:')


# new testing objects
ind_test = np.array([10, 20, 50, 100, 120, 150, 200, 220, 250, 300, 320, 350, 400, 420]) #np.random.choice(np.arange(fluxes.shape[0]), (N_new,))
Y_new_test = np.zeros((len(ind_test), L))
Z_new_test = np.zeros((len(ind_test), Q))
N_new = len(ind_test)

X_new = fluxes[ind_test, wl_start:wl_start+D] - X_mean
X_ivar_new = ivars[ind_test, wl_start:wl_start+D]

chi2 = Chi2_Matrix(X, 1./X_var, X_new, X_ivar_new)
all_NN = np.zeros((len(ind_test), L))


for i in range(N_new):
    
    # starting_guess
    y0, index_n = NN(i, chi2, Y)
    z0 = Z_final[index_n, :]
    all_NN[i, :] = y0
    
    Z_new_n, Y_new_n, success_z, success_y = predictX(X_new[i, :], 1./X_ivar_new[i, :], X, X_var, Y, Y_var, Z_final, hyper_params, y0, z0)
    Y_new_test[i, :] = Y_new_n
    Z_new_test[i, :] = Z_new_n
    
    j = 0
    mean_new, var_new, foo = mean_var(Z_final, Z_new_n, X, X_var, kernel1, theta_rbf, theta_band)
    chi2_test = np.sum(((X_new[i, :] - mean_new)**2) * X_ivar_new[i, :])
    plt.figure(figsize=(8, 6))
    plt.tick_params(axis=u'both', direction='in', which='both')
    plt.plot(wl[wl_start:wl_start+D], X_new[i, :] + X_mean, label='original data', color='k')
    plt.fill_between(wl[wl_start:wl_start+D], X_new[i, :] + X_mean - 0.5*np.sqrt(1./X_ivar_new[i, :]), X_new[i, :] + X_mean + 0.5*np.sqrt(1./X_ivar_new[i, :]), color='k', alpha = .3)
    plt.plot(wl[wl_start:wl_start+D], mean_new + X_mean, label=r'GPLVM model' + '\n $\chi^2\sim{}$'.format(round(chi2_test, 4)), color='r')
    plt.fill_between(wl[wl_start:wl_start+D], mean_new + X_mean - 0.5*np.sqrt(var_new), mean_new + X_mean + 0.5*np.sqrt(var_new), color='r', alpha = .3)
    plt.title(r'star {0}: $\theta_{{\rm band}} = {1},\,\theta_{{\rm rbf}} = {2},\, \gamma_{{\rm band}} = {3},\,\gamma_{{\rm rbf}} = {4}$, opt.: {5}'.format(ind_test[i], theta_band, theta_rbf, gamma_band, gamma_rbf, success_z), fontsize = 12)
    plt.legend(frameon = True)
    plt.xlabel('wavelength', fontsize = fsize)
    plt.ylabel('data', fontsize = fsize)
    plt.ylim(0, 1.2)
    plt.savefig('plots/{0}/testing_object_{1}_{2}.png'.format(date, name, ind_test[i]))
    plt.close()


# testing labels    
Y_old = tr_label_input[ind_test, :]
Y_new_rescaled = np.zeros_like(Y_new_test)
all_NN_rescaled = np.zeros_like(all_NN)

for l in range(L):
    Y_new_rescaled[:, l] = Y_new_test[:, l] * scales[l] + pivots[l]
    all_NN_rescaled[:, l] = all_NN[:, l] * scales[l] + pivots[l]

for i, l in enumerate(labels):

    orig = Y_old[:, i]
    gp_values = Y_new_rescaled[:, i]
    scatter = np.round(np.std(orig - gp_values), 5)
    bias = np.round(np.mean(orig - gp_values), 5)    
    scatter_nn = np.round(np.std(orig - all_NN_rescaled[:, i]), 5)
    bias_nn = np.round(np.mean(orig - all_NN_rescaled[:, i]), 5)
    
    xx = [-10000, 10000]
    plt.figure(figsize=(6, 6))
    plt.scatter(orig, gp_values, color=colors[-2], label=' bias = {0} \n scatter = {1}'.format(bias, scatter), marker = 'o')
    plt.scatter(orig, all_NN_rescaled[:, i], color=colors[0], label=' NN: bias = {0} \n scatter = {1}'.format(bias_nn, scatter_nn), marker = 'o')
    plt.plot(xx, xx, color=colors[2], linestyle='--')
    plt.xlabel(r'reference labels {}'.format(latex[l]), size=fsize)
    plt.ylabel(r'inferred values {}'.format(latex[l]), size=fsize)
    plt.tick_params(axis=u'both', direction='in', which='both')
    plt.xlim(plot_limits[l])
    plt.ylim(plot_limits[l])
    plt.tight_layout()
    plt.legend(loc=2, fontsize=14, frameon=True)
    plt.title('#stars: {0}, #pixels: {1}, $\\theta_{{band}} = {4}$, $\\theta_{{rbf}} = {5}$, $\gamma_{{band}} = {6}$, $\gamma_{{rbf}} = {7}$'.format(len(ind_test), D, L, Q, theta_band, theta_rbf, gamma_band, gamma_rbf), fontsize=fsize)
    plt.savefig('plots/{0}/1to1_test_{1}_{2}.png'.format(date, l, name))
    plt.close()
    
# calculate distribution of chi^2 for all spectra... (or 100 randomly chosen ones...)
#np.random.seed(10)
#chis = []
#for i in range(100):
#    ind_test_i = np.random.choice(np.arange(len(fluxes)), (N_new,)) #np.array([ind_test[i]])
#    print ind_test_i
#    X_new = fluxes[ind_test_i, wl_start:wl_start+D] - X_mean
#    X_new_var = 1./ivars[ind_test_i, wl_start:wl_start+D]    
#    Z_new, Y_new, success_z, success_y = predictX(N_new, X_new, X_new_var, X, X_var, Y, Y_var, Z_final, hyper_params)    
#    j = 0
#    mean_new, var_new, foo = mean_var(Z_final, Z_new[j, :], X, X_var, kernel1, theta_rbf, theta_band)
#    chi2_test = np.sum((X_new[j, :] - mean_new)**2/X_new_var[j, :])
#    chis.append(chi2_test)
#plt.hist(chis, label = r'$\langle \chi^2\rangle = {0}$'.format(np.median(chis)))
#plt.tick_params(axis=u'both', direction='in', which='both')
#plt.xlabel(r'$\chi^2$')
#plt.ylabel('counts')
#plt.legend(frameon=True)
#plt.title(r'$\theta_{{rbf}} = {0}, \theta_{{band}} = {1}, \gamma_{{rbf}} = {2}, \gamma_{{band}} = {3}$'.format(theta_rbf, theta_band, gamma_rbf, gamma_band))
#plt.savefig('plots/{0}/chis_{1}.png'.format(date, name))


# -------------------------------------------------------------------------------
# xxx
# -------------------------------------------------------------------------------'''


