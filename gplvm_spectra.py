#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:58:25 2017

@author: eilers
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import scipy.optimize as op
import seaborn as sns
import time
import pickle
from astropy.table import Column, Table, join, vstack
import sys
from astropy.io import fits
#import schwimmbad 

from functions_gplvm import lnL_Z_old, lnL_h, mean_var, kernelRBF, predictX, make_label_input, get_pivots_and_scales, PCAInitial
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

## -------------------------------------------------------------------------------
## load spectra and labels 
## -------------------------------------------------------------------------------
#
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

## -------------------------------------------------------------------------------
## load spectra and labels (cluster stars)
## -------------------------------------------------------------------------------
##
## The APOGEE data file has more labels missing than the other file!!! 
## QUESTION: WHERE SHOULD WE TAKE STELLAR LABELS FROM?!
##
##create_data_set = False
##
##if create_data_set:
##    table = Table.read('data/meszaros13_table4_mod_new.txt', format='ascii', data_start = 0) 
##    table['col3'].name = 'CLUSTER'
##    table['col2'].name = 'APOGEE_ID'
##    
##    # open APOGEE data
##    apogee = Table.read('./data/allStar-l31c.2.fits', format='fits', hdu = 1)
##    joined_data = join(table, apogee, keys = 'APOGEE_ID', join_type='left')
##    
##    foo, idx = np.unique(joined_data['APOGEE_ID'], return_index = True)
##    training_labels = joined_data[idx]
##    f = open('data/train_labels_cluster.pickle', 'w')  
##    pickle.dump(training_labels, f)
##    f.close()
##
##f = open('data/train_labels_cluster.pickle', 'r')    
##training_labels = pickle.load(f)
##f.close()
#
#table = Table.read('data/meszaros13_table4_mod_new.txt', format='ascii', data_start = 0) 
#table['col3'].name = 'CLUSTER'
#
#f = open('data/train_labels_feh.pickle', 'r')    
#training_labels = pickle.load(f)
#f.close()
#training_labels = Table(training_labels)
#training_labels['col0'].name = 'TEFF'
#training_labels['col1'].name = 'TEFF_ERR'
#training_labels['col2'].name = 'LOGG'
#training_labels['col3'].name = 'LOGG_ERR'
#training_labels['col4'].name = 'FE_H'
#training_labels['col5'].name = 'FE_H_ERR'
#
#f = open('data/all_data_norm.pickle', 'r')    
#spectra = pickle.load(f)
#f.close()
#wl = spectra[:, 0, 0]
#fluxes = spectra[:, :, 1].T
#ivars = (1./(spectra[:, :, 2]**2)).T  
#        
## mask all spectra with a -9999.9 entry! (all spectra have [alpha/Fe] measured)
#missing = np.logical_or(training_labels['TEFF'] < -9000., training_labels['LOGG'] < -9000., training_labels['FE_H'] < -9000.)
#training_labels = training_labels[~missing]
#fluxes = fluxes[~missing]
#ivars = ivars[~missing]
#table = table[~missing]
#        
## exclude pleiades
#pleiades = table['CLUSTER'] == 'Pleiades'
#training_labels = training_labels[~pleiades]
#fluxes = fluxes[~pleiades]
#ivars = ivars[~pleiades]

## -------------------------------------------------------------------------------
## # calculate K_MAG_ABS and Q
## -------------------------------------------------------------------------------
#
#Q = 10**(0.2*training_labels['K']) * training_labels['parallax']/100.                    # assumes parallaxes is in mas
#Q_err = training_labels['parallax_error'] * 10**(0.2*training_labels['K'])/100. 
#Q = Column(Q, name = 'Q_MAG')
#Q_err = Column(Q_err, name = 'Q_MAG_ERR')
#training_labels.add_column(Q, index = 12)
#training_labels.add_column(Q_err, index = 13)

# -------------------------------------------------------------------------------
# load spectra and labels (giant stars)
# -------------------------------------------------------------------------------

#f = open('data/training_labels_apogee_tgas_giants.pickle', 'r')    
#training_labels = pickle.load(f)
#f.close()

hdulist = fits.open('data/training_labels_apogee_tgas_giants.fits')
training_labels = hdulist[1].data
       
f = open('data/apogee_spectra_norm_giants.pickle', 'r')    
spectra = pickle.load(f)
f.close()

wl = spectra[:, 0, 0]
fluxes = spectra[:, :, 1].T
ivars = (1./(spectra[:, :, 2]**2)).T 
        
# add pixel mask to remove gaps between chips! Otherwise code fails because all pixels are being masked in the gaps!
gaps = (np.sum(fluxes, axis = 0)) == float(fluxes.shape[0])
wl = wl[~gaps]
fluxes = fluxes[:, ~gaps]
ivars = ivars[:, ~gaps]

# -------------------------------------------------------------------------------
# load spectra and labels (validation set)
# -------------------------------------------------------------------------------
                                    
#f = open('data/training_labels_apogee_tgas_validation.pickle', 'r')    
#training_labels_validation = pickle.load(f)
#f.close()

hdulist = fits.open('data/training_labels_apogee_tgas_validation.fits')
training_labels_validation = hdulist[1].data
                                    
good = training_labels_validation['parallax']/training_labels_validation['parallax_error'] > 9
training_labels_validation = training_labels_validation[good]

f = open('data/apogee_spectra_norm_validation.pickle', 'r')    
spectra_validation = pickle.load(f)
f.close() 

spectra_validation = spectra_validation[:, good, :]                

fluxes_validation = spectra_validation[:, :, 1].T
ivars_validation = (1./(spectra_validation[:, :, 2]**2)).T 

fluxes_validation = fluxes_validation[:, ~gaps]
ivars_validation = ivars_validation[:, ~gaps]

all_fluxes = np.vstack([fluxes, fluxes_validation])
all_ivars = np.vstack([ivars, ivars_validation])

# -------------------------------------------------------------------------------
# get validation set from 10% of training set
# -------------------------------------------------------------------------------

training_labels = Table(training_labels)
training_labels_validation = Table(training_labels_validation)
all_training_labels = vstack([training_labels, training_labels_validation])
all_ind = np.arange(len(all_training_labels))

# training set contains 90% of data
N = int(len(all_training_labels) - np.ceil(len(all_training_labels) / 10.))

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
latex["Q_K"] = r"$Q_K$"

plot_limits = {}
plot_limits['TEFF'] = (3000, 7000)
plot_limits['FE_H'] = (-2.5, 1)
plot_limits['LOGG'] = (0, 4.)
plot_limits['ALPHA_FE'] = (-.2, .6)
plot_limits['KMAG_ABS'] = (-1, -6)
plot_limits['Q_K'] = (0, 1)

labels = np.array(['TEFF', 'LOGG', 'FE_H']) #, 'Q_K', 'ALPHA_M', 'Q_MAG', 'N_FE', 'C_FE'])
Nlabels = len(labels)
latex_labels = [latex[l] for l in labels]
tr_label_input, tr_var_input = make_label_input(labels, all_training_labels)
print(Nlabels, tr_label_input.shape, tr_var_input.shape, fluxes.shape, ivars.shape) 

# -------------------------------------------------------------------------------
# take random indices
# -------------------------------------------------------------------------------

np.random.seed(34)  # 34
indices = np.arange(all_fluxes.shape[0])
np.random.shuffle(indices)
ind_train = indices[:N]
ind_validation = indices[N:]

#tr_label_validation, tr_var_validation = make_label_input(labels, training_labels_validation)

pivots, scales = get_pivots_and_scales(tr_label_input[ind_train])
tr_label_input_scaled = (tr_label_input[ind_train] - pivots[None, :]) / scales[None, :]
tr_var_input_scaled = tr_var_input[ind_train] / (scales[None, :]**2)
print('pivots: {}'.format(pivots))
print('scales: {}'.format(scales))

# -------------------------------------------------------------------------------
# constants
# -------------------------------------------------------------------------------

#N = len(tr_label_input[ind_train])         # number of stars
D = len(wl)                             # number of pixels
Q = 10                                       # number latent dimensions, Q<=L
L = len(labels)                             # number of labels
pixel = np.arange(D)
wl_start = 0  

theta_rbf = 0.1 * np.ones((D, ))
gamma_rbf = 1. * np.ones((L, )) 
theta_band, gamma_band = 1e-4, 0.01 

theta_rbf_name = theta_rbf[0]
gamma_rbf_name = gamma_rbf[0]

name = '{0}_{1}_{2}_{3}_Q{4}_D{5}_N{6}_L{7}_noQ_onethirdfehmissing'.format(theta_rbf_name, theta_band, gamma_rbf_name, gamma_band, Q, D, N, L)
date = 'missing_labels' # 'Q_tests' #
print name

# -------------------------------------------------------------------------------
# plots for training and validation set 
# -------------------------------------------------------------------------------

#cm = plt.cm.get_cmap('viridis')
#sc = plt.scatter(all_training_labels['TEFF'][ind_train], all_training_labels['LOGG'][ind_train], c = all_training_labels['FE_H'][ind_train], vmin = -2., vmax = .5, cmap = cm)
#cbar = plt.colorbar(sc)
#cbar.set_label(r'$[Fe/H]$', rotation=270, size=12, labelpad = 10)
#plt.xlim(5700, 3700)
#plt.xlabel(r'$T_{eff}$')
#plt.ylabel(r'$\log g$')
#plt.ylim(4, 0)
#plt.tick_params(axis=u'both', direction='in', which='both')
#plt.title(r'training set: ${}$ stars'.format(len(tr_label_input[ind_train])))
#plt.savefig('plots/{0}/teff_logg_train_s28.png'.format(date))
#plt.close()
#
#sc = plt.scatter(all_training_labels['TEFF'][ind_train], all_training_labels['Q_K'][ind_train], c = all_training_labels['A_K'][ind_train], vmin = 0., vmax = .1, cmap = cm)
#cbar = plt.colorbar(sc)
#cbar.set_label(r'$A_K$', rotation=270, size=12, labelpad = 10)
#plt.xlim(5700, 3700)
#plt.ylim(1., 0.)
#plt.xlabel(r'$T_{eff}$')
#plt.ylabel(r'$Q_{K, corr}$')
#plt.tick_params(axis=u'both', direction='in', which='both')
#plt.title(r'training set: ${}$ stars'.format(len(tr_label_input[ind_train])))
#plt.savefig('plots/{0}/teff_QK_colorA_train_s28.png'.format(date))
#plt.close()
#
#sc = plt.scatter(all_training_labels['TEFF'][ind_validation], all_training_labels['LOGG'][ind_validation], c = all_training_labels['FE_H'][ind_validation], vmin = -2., vmax = .5, cmap = cm)
#cbar = plt.colorbar(sc)
#cbar.set_label(r'$[Fe/H]$', rotation=270, size=12, labelpad = 10)
#plt.xlim(5700, 3700)
#plt.xlabel(r'$T_{eff}$')
#plt.ylabel(r'$\log g$')
#plt.ylim(4, 0)
#plt.tick_params(axis=u'both', direction='in', which='both')
#plt.title(r'validation set: ${}$ stars'.format(len(tr_label_input[ind_validation])))
#plt.savefig('plots/{0}/teff_logg_val_s28.png'.format(date))
#plt.close()
#
#plt.figure()
#sc = plt.scatter(all_training_labels['TEFF'][ind_validation], all_training_labels['Q_K'][ind_validation], c = all_training_labels['A_K'][ind_validation], vmin = 0., vmax = .1, cmap = cm)
#cbar = plt.colorbar(sc)
#cbar.set_label(r'$A_K$', rotation=270, size=12, labelpad = 10)
#plt.xlim(5700, 3700)
#plt.ylim(1., 0.)
#plt.xlabel(r'$T_{eff}$')
#plt.ylabel(r'$Q_{K, corr}$')
#plt.tick_params(axis=u'both', direction='in', which='both')
#plt.title(r'validation set: ${}$ stars'.format(len(tr_label_input[ind_validation])))
#plt.savefig('plots/{0}/teff_QK_colorA_val_s28.png'.format(date))
#plt.close()

# -------------------------------------------------------------------------------
# input data
# -------------------------------------------------------------------------------

# subtract median spectrum from fluxes
X_orig = all_fluxes[ind_train, wl_start:wl_start+D] 
X_mean = np.median(X_orig, axis = 0)        
X = X_orig - X_mean

# plot the mean!          
          
X_var = 1./all_ivars[ind_train, wl_start:wl_start+D]
Y = tr_label_input_scaled#[ind_train, :]
Y_var = tr_var_input_scaled#[ind_train, :]

# -------------------------------------------------------------------------------
# hyper vectors
# -------------------------------------------------------------------------------

#theta_rbf = 1. * np.std(X, axis = 0) # for whatever reason
#gamma_rbf = 1. * np.std(Y, axis = 0) # for whatever reason

# -------------------------------------------------------------------------------
# contrsuct NxD boolean object which is True, if data is good. 
# -------------------------------------------------------------------------------

#N = N*2/3
X = X[:N, :]
Y = Y[:N, :]
X_var = X_var[:N, :]
Y_var = Y_var[:N, :]


X_mask = np.ones((N, D), dtype = bool)
for n in range(N):
    for d in range(D):
        if 1./X_var[n, d] < 10.:
            X_mask[n, d] = False
#            X_ivar[n, d] = 0.

Y_mask = np.ones((N, L), dtype = bool)
for n in range(N):
    for l in range(L):
        if 1./Y_var[n, l] < 1.:
            Y_mask[n, l] = False
#            Y_ivar[n, l] = 0.
                       
#good_labels[:, 0] = False
#good_labels[10, 1] = False
#Y_mask[:N/3, 0] = False
#Y_mask[N/3:N*2/3, 1] = False      
Y_mask[N*2/3:, 2] = False
      
plt.hist(np.sum(X_mask, axis = 0), bins = 50)
#plt.yscale('log')
plt.xlabel('# of missing pixels')
plt.tick_params(axis=u'both', direction='in', which='both')
plt.savefig('plots/{0}/bad_pixels_D{1}_N{2}.png'.format(date, D, N))
plt.close()

plt.hist(np.sum(Y_mask, axis = 1))
plt.xlabel('# of missing labels')
plt.xlim(0, L)
#plt.yscale('log')
plt.tick_params(axis=u'both', direction='in', which='both')
plt.savefig('plots/{0}/bad_labels_D{1}_N{2}.png'.format(date, D, N))
plt.close()


# -------------------------------------------------------------------------------
# initialize parameters
# -------------------------------------------------------------------------------

hyper_params = np.array([theta_rbf, theta_band, gamma_rbf, gamma_band])

#Z_initial = np.zeros((N, Q)) 
#Z_initial[:, :L] = Y[:]                # THIS DOESN'T WORK IF Q > L! ALL MISSING DIMENSIONS STAY 0!!
Z_initial = PCAInitial(X, Q)            # PCA initialize, use mean as first component, i.e. 1 as initial guess for Z_initial for each star
Z = np.reshape(Z_initial, (N*Q,))

# -------------------------------------------------------------------------------
# optimization
# -------------------------------------------------------------------------------

print('initial hyper parameters: %s' %hyper_params)
#print('initial latent variables: %s' %Z)

#pool = schwimmbad.SerialPool()
#cygnet_likelihood = CygnetLikelihood(X, Y, Q, hyper_params, X_var, Y_var, X_mask, Y_mask)

t1 = time.time()

max_iter = 1
for t in range(max_iter):
    
#    # optimize hyperparameters
#    print("optimizing hyper parameters")
#    res = op.minimize(lnL_h, x0 = hyper_params, args = (X, Y, Z, Z_initial, X_var, Y_var, X_mask, Y_mask), method = 'L-BFGS-B', jac = True, 
#                      options={'gtol':1e-12, 'ftol':1e-12})
#    
#    # update hyperparameters
#    hyper_params = res.x
#    print(res)
#    print('success: {}'.format(res.success))
#    print('new hyper parameters: {}'.format(res.x))
    
    # optimize Z
    print("optimizing latent parameters")
    res = op.minimize(lnL_Z_old, x0 = Z, args = (X, Y, hyper_params, Z_initial, X_var, Y_var, X_mask, Y_mask), method = 'L-BFGS-B', jac = True, 
                      options={'gtol':1e-12, 'ftol':1e-12})
#    res = op.minimize(cygnet_likelihood, x0 = Z, args=(pool,), method = 'L-BFGS-B', jac = True, 
#                      options={'gtol':1e-12, 'ftol':1e-12})
             
    # update Z
    Z = res.x
    print(res)
    print('success: {}'.format(res.success))
    # print('new Z: {}'.format(res.x))


#sys.exit(0)
    
t2 = time.time()
print('optimization in {} s.'.format(t2-t1))

Z_final = np.reshape(Z, (N, Q))
#kernel1 = kernelRBF(Z_final, theta_rbf, theta_band)
#kernel2 = kernelRBF(Z_final, gamma_rbf, theta_band)

# -------------------------------------------------------------------------------
# inferring new labels of training objects  
# -------------------------------------------------------------------------------

mean_labels = np.zeros_like(Y)
for j in range(N):   
    mean, var, foo = mean_var(Z_final, Z_final[j, :], Y, Y_var, gamma_rbf, gamma_band)
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
    plt.title(r'#stars: {0}, #pixels: {1}, #labels: {2}, #latent dim.: {3}, $\theta_{{band}} = {4}$, $\theta_{{rbf}} = {5}$, $\gamma_{{band}} = {6}$, $\gamma_{{rbf}} = {7}$'.format(N, D, L, Q, theta_band, theta_rbf_name, gamma_band, gamma_rbf_name), fontsize=12)
    plt.tight_layout()
    plt.savefig('plots/{0}/Latent2Label_{1}_{2}.png'.format(date, l, name))
    plt.close()

#q = 0
#plt.figure(figsize=(7, 7))
#plt.tick_params(axis=u'both', direction='in', which='both')
#cm = plt.cm.get_cmap('viridis')
#sc = plt.scatter(tr_label_input[ind_train, q], Z_final[:, q], marker = 'o', cmap = cm)
##cbar = plt.colorbar(sc)
##cbar.set_label(r'{}'.format(latex[l]), rotation=270, size=fsize, labelpad = 10)
#plt.xlim(0, .7)
##plt.ylim(0, .7)
#plt.ylabel(r'latent dimension {}'.format(q), fontsize = fsize)
#plt.xlabel(r'$Q_K$', fontsize = fsize)
#plt.title(r'#stars: {0}, #pixels: {1}, #labels: {2}, #latent dim.: {3}'.format(N, D, L, Q, theta_band, theta_rbf, gamma_band, gamma_rbf), fontsize=12)
#plt.tight_layout()
#plt.savefig('plots/{0}/Latent2Label_{1}_{2}.png'.format(date, l, name))


# data and model for training object
j = 2
mean, var, foo = mean_var(Z_final, Z_final[j, :], X, X_var, theta_rbf, theta_band)
chi2 = np.sum((X[j, :] - mean)**2/X_var[j, :])
#plt.figure(figsize=(8, 6))
fig, ax = plt.subplots(ncols = 2, nrows = 1, figsize = (16, 6))
plt.title(r'star {0}: $\theta_{{\rm band}} = {1},\,\theta_{{\rm rbf}} = {2}$'.format(ind_train[j], theta_band, theta_rbf_name), fontsize = fsize)
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
    chi2_label = np.round(np.sum((orig - gp_values)**2 /tr_var_input[ind_train[:N], i]), 4)
    
    xx = [-10000, 10000]
    plt.figure(figsize=(6, 6))
    plt.scatter(orig, gp_values, color=colors[-2], label=' bias = {0} \n scatter = {1} \n $\chi^2$ = {2}'.format(bias, scatter, chi2_label), marker = 'o')
    plt.plot(xx, xx, color=colors[2], linestyle='--')
    plt.xlabel(r'reference labels {}'.format(latex[l]), size=fsize)
    plt.ylabel(r'inferred values {}'.format(latex[l]), size=fsize)
    plt.tick_params(axis=u'both', direction='in', which='both')
    plt.xlim(plot_limits[l])
    plt.ylim(plot_limits[l])
    plt.tight_layout()
    plt.legend(loc=2, fontsize=14, frameon=True)
    plt.title('#stars: {0}, #pixels: {1}, $\\theta_{{band}} = {4}$, $\\theta_{{rbf}} = {5}$, $\gamma_{{band}} = {6}$, $\gamma_{{rbf}} = {7}$'.format(N, D, L, Q, theta_band, theta_rbf_name, gamma_band, gamma_rbf_name), fontsize=12)
    plt.savefig('plots/{0}/1to1_{1}_{2}.png'.format(date, l, name, t))
    plt.close()

# -------------------------------------------------------------------------------
# prediction for new test object
# -------------------------------------------------------------------------------
    
print('prediction of test object:')

#ind_test = np.arange(fluxes_validation.shape[0])
Y_new_test = np.zeros((len(ind_validation), L))
Z_new_test = np.zeros((len(ind_validation), Q))
N_new = len(ind_validation)

X_new = all_fluxes[ind_validation, wl_start:wl_start+D] - X_mean
X_ivar_new = all_ivars[ind_validation, wl_start:wl_start+D]

X_mask_new = np.ones((N_new, D), dtype = bool)
for n in range(N_new):
    for d in range(D):
        if X_ivar_new[n, d] < 10.:
            X_mask_new[n, d] = False

chi2 = Chi2_Matrix(X, 1./X_var, X_new, X_ivar_new)
all_NN = np.zeros((len(ind_validation), L))

all_chis = []

for i in range(N_new):
    
    # starting_guess
    y0, index_n = NN(i, chi2, Y)
    z0 = Z_final[index_n, :]
    all_NN[i, :] = y0
    
    good_stars_new = X_mask_new[i, :]
    Z_new_n, Y_new_n, success_z, success_y = predictX(X_new[i, good_stars_new], 1./X_ivar_new[i, good_stars_new], X, X_var, Y, Y_var, Z_final, hyper_params, y0, z0)
    Y_new_test[i, :] = Y_new_n
    Z_new_test[i, :] = Z_new_n
    
    j = 0
    mean_new, var_new, foo = mean_var(Z_final, Z_new_n, X, X_var, theta_rbf, theta_band)
    chi2_test = np.sum(((X_new[i, :] - mean_new)**2) * X_ivar_new[i, :])
    all_chis.append(chi2_test)
    plt.figure(figsize=(8, 6))
    plt.tick_params(axis=u'both', direction='in', which='both')
    plt.plot(wl[wl_start:wl_start+D], X_new[i, :] + X_mean, label='original data', color='k')
    plt.fill_between(wl[wl_start:wl_start+D], X_new[i, :] + X_mean - 0.5*np.sqrt(1./X_ivar_new[i, :]), X_new[i, :] + X_mean + 0.5*np.sqrt(1./X_ivar_new[i, :]), color='k', alpha = .3)
    plt.plot(wl[wl_start:wl_start+D], mean_new + X_mean, label=r'GPLVM model' + '\n $\chi^2\sim{}$'.format(round(chi2_test, 4)), color='r')
    plt.fill_between(wl[wl_start:wl_start+D], mean_new + X_mean - 0.5*np.sqrt(var_new), mean_new + X_mean + 0.5*np.sqrt(var_new), color='r', alpha = .3)
    plt.title(r'star {0}: $\theta_{{\rm band}} = {1},\,\theta_{{\rm rbf}} = {2},\, \gamma_{{\rm band}} = {3},\,\gamma_{{\rm rbf}} = {4}$, opt.: {5}'.format(ind_validation[i], theta_band, theta_rbf_name, gamma_band, gamma_rbf_name, success_z), fontsize = 12)
    plt.legend(frameon = True)
    plt.xlabel('wavelength', fontsize = fsize)
    plt.ylabel('data', fontsize = fsize)
    plt.ylim(0, 1.2)
    plt.savefig('plots/{0}/testing_object_{1}_{2}.png'.format(date, name, ind_validation[i]))
    plt.close()

plt.hist(all_chis, label = r'$\langle\chi^2\rangle = {}$'.format(round(np.median(all_chis), 2)))
plt.title(r'# of stars: {0}, # of pixels: {5}, $\theta_{{\rm band}} = {1},\,\theta_{{\rm rbf}} = {2},\, \gamma_{{\rm band}} = {3},\,\gamma_{{\rm rbf}} = {4}$'.format(len(ind_validation), theta_band, theta_rbf_name, gamma_band, gamma_rbf_name, D), fontsize = 12)
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel(r'$\chi^2$', fontsize = fsize)
plt.legend(frameon = True)
plt.savefig('plots/{0}/all_chis_{1}.png'.format(date, name))
plt.close()

# testing labels    
Y_old = tr_label_input[ind_validation, :]
Y_new_rescaled = np.zeros_like(Y_new_test)
all_NN_rescaled = np.zeros_like(all_NN)

for l in range(L):
    Y_new_rescaled[:, l] = Y_new_test[:, l] * scales[l] + pivots[l]
    all_NN_rescaled[:, l] = all_NN[:, l] * scales[l] + pivots[l]

for i, l in enumerate(labels):

    orig = Y_old[:, i]
    gp_values = Y_new_rescaled[:, i]
    scatter = np.round(np.std(orig - gp_values), 4)
    bias = np.round(np.mean(orig - gp_values), 4)    
    #scatter_nn = np.round(np.std(orig - all_NN_rescaled[:, i]), 5)
    #bias_nn = np.round(np.mean(orig - all_NN_rescaled[:, i]), 5)    
    chi2_label = np.round(np.sum((orig - gp_values)**2 /tr_var_input[ind_validation, i]), 4)
    
    xx = [-10000, 10000]
    plt.figure(figsize=(6, 6))
    plt.scatter(orig, gp_values, color=colors[-2], label=' bias = {0} \n scatter = {1} \n $\chi^2$ = {2}'.format(bias, scatter, chi2_label), marker = 'o')
    #plt.scatter(orig, all_NN_rescaled[:, i], color=colors[0], label=' NN: bias = {0} \n scatter = {1}'.format(bias_nn, scatter_nn), marker = 'o')
    plt.plot(xx, xx, color=colors[2], linestyle='--')
    plt.xlabel(r'reference labels {}'.format(latex[l]), size=fsize)
    plt.ylabel(r'inferred values {}'.format(latex[l]), size=fsize)
    plt.tick_params(axis=u'both', direction='in', which='both')
    plt.xlim(plot_limits[l])
    plt.ylim(plot_limits[l])
    plt.tight_layout()
    plt.legend(loc=2, fontsize=14, frameon=True)
    plt.title('#stars: {0}, #pixels: {1}, $\\theta_{{band}} = {4}$, $\\theta_{{rbf}} = {5}$, $\gamma_{{band}} = {6}$, $\gamma_{{rbf}} = {7}$'.format(len(ind_validation), D, L, Q, theta_band, theta_rbf_name, gamma_band, gamma_rbf_name), fontsize=11)
    plt.savefig('plots/{0}/1to1_test_{1}_{2}.png'.format(date, l, name))
    plt.close()
    
    if l == 'Q_K' or l == 'Q_L':
        orig = 5. * (np.log10(Y_old[:, i]))
        gp_values = 5. * (np.log10(Y_new_rescaled[:, i]))
        scatter = np.round(np.std(orig - gp_values), 4)
        bias = np.round(np.mean(orig - gp_values), 4)    
        if l == 'Q_K':
            chi2_label = np.round(np.sum((orig - gp_values)**2 / all_training_labels['K_ERR'][ind_validation]), 4)
        elif l == 'Q_L':
            chi2_label = np.round(np.sum((orig - gp_values)**2 / all_training_labels['L_ERR'][ind_validation]), 4)        
        xx = [-10000, 10000]
        plt.figure(figsize=(6, 6))
        plt.scatter(orig, gp_values, color=colors[-2], label=' bias = {0} \n scatter = {1} \n $\chi^2$ = {2}'.format(bias, scatter, chi2_label), marker = 'o')
        #plt.scatter(orig, all_NN_rescaled[:, i], color=colors[0], label=' NN: bias = {0} \n scatter = {1}'.format(bias_nn, scatter_nn), marker = 'o')
        plt.plot(xx, xx, color=colors[2], linestyle='--')
        if l == 'Q_K':
            plt.xlabel(r'reference labels $M_{K}$', size=fsize)
            plt.ylabel(r'inferred values $M_{K}$', size=fsize)
        elif l == 'Q_L':
            plt.xlabel(r'reference labels $M_{L}$', size=fsize)
            plt.ylabel(r'inferred values $M_{L}$', size=fsize)            
        plt.tick_params(axis=u'both', direction='in', which='both')
        plt.xlim(-5, 0)
        plt.ylim(-5, 0)
        plt.tight_layout()
        plt.legend(loc=2, fontsize=14, frameon=True)
        plt.title('#stars: {0}, #pixels: {1}, $\\theta_{{band}} = {4}$, $\\theta_{{rbf}} = {5}$, $\gamma_{{band}} = {6}$, $\gamma_{{rbf}} = {7}$'.format(len(ind_validation), D, L, Q, theta_band, theta_rbf_name, gamma_band, gamma_rbf_name), fontsize=11)
        if l == 'Q_K':        
            plt.savefig('plots/{0}/1to1_test_{1}_{2}.png'.format(date, 'M_K', name))
        elif l == 'Q_L':        
            plt.savefig('plots/{0}/1to1_test_{1}_{2}.png'.format(date, 'M_L', name))
            plt.close()


# -------------------------------------------------------------------------------
# mark RC stars
# -------------------------------------------------------------------------------'''

#hdulist = fits.open('data/RGB.fits')
#lab_rgb = hdulist[1].data
#lab_rgb = Table(lab_rgb)
#
#
#Ntrain = len(all_training_labels[ind_train])
#
#rgb = np.ones((Ntrain, ), dtype = bool)
#
#for i in range(len(lab_rgb)):
#    for j in range(Ntrain):
#        if lab_rgb['APOGEE_ID'][i] == all_training_labels['APOGEE_ID'][j]:
#            rgb[i] = False
#
#from matplotlib.colors import ListedColormap
#
#cm = ListedColormap([colors[0], colors[5]])
#
## latent space color coded by labels
#l = 'Q_K'
#q = 0
#fig, ax = plt.subplots(figsize=(9, 6))
#plt.tick_params(axis=u'both', direction='in', which='both')
#sc = plt.scatter(Z_final[:, q], Z_final[:, q+1], c = rgb, marker = 'o', cmap = cm, vmin = 0, vmax = 1)
#cbar = plt.colorbar(sc)
##cbar.set_label(r'RC', rotation=270, size=fsize, labelpad = 10)
#cbar.ax.set_yticklabels(['', 'RGB', '', '', 'RC'])
##plt.xlim(-4, 4)
##plt.ylim(-4, 4)
#plt.xlabel(r'latent dimension {}'.format(q), fontsize = fsize)
#plt.ylabel(r'latent dimension {}'.format(q+1), fontsize = fsize)
#plt.title(r'#stars: {0}, #pixels: {1}, #labels: {2}, #latent dim.: {3}, $\theta_{{band}} = {4}$, $\theta_{{rbf}} = {5}$, $\gamma_{{band}} = {6}$, $\gamma_{{rbf}} = {7}$'.format(N, D, L, Q, theta_band, theta_rbf_name, gamma_band, gamma_rbf_name), fontsize=12)
#plt.tight_layout()
#plt.savefig('plots/{0}/Latent2Label_{1}_{2}_RC.png'.format(date, l, name))
#plt.close()

# -------------------------------------------------------------------------------
# xxx
# -------------------------------------------------------------------------------'''

