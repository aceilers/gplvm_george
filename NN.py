#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 11:04:16 2017

@author: eilers
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
from astropy.table import Table, Column, vstack
import seaborn as sns
from astropy.io import fits

from functions_gplvm import make_label_input

colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple", "pale red", "orange", "red", "blue"]
colors = sns.xkcd_palette(colors)
lsize = 14
sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white', 'axes.edgecolor':'black', 'xtick.direction': 'in', 'ytick.direction': 'in'})
sns.set_style("ticks")
matplotlib.rcParams['ytick.labelsize'] = lsize
matplotlib.rcParams['xtick.labelsize'] = lsize

# -------------------------------------------------------------------------------
# functions
# -------------------------------------------------------------------------------

def Chi2_Matrix(Y1, Y1_ivar, Y2, Y2_ivar, infinite_diagonal = False):
    """
    Returns N1 x N2 matrix of chi-squared values.
    A clever user will decide on the Y1 vs Y2 choice sensibly.
    """
    N1, D = Y1.shape
    N2, D2 = Y2.shape
    assert D == D2
    assert Y1_ivar.shape == Y1.shape
    assert Y2_ivar.shape == Y2.shape
    chi2 = np.zeros((N1, N2)) 
#    # uses a loop to save memory
#    for n1 in range(N1):
#        DY = Y2 - Y1[n1, :]
#        denominator = Y2_ivar + Y1_ivar[n1, :]
#        ivar = Y2_ivar * Y1_ivar[n1, :] / (denominator + (denominator <= 0.))
#        chi2[n1, :] = np.dot(ivar * DY, DY.T)
    for n1 in range(N1):
        for n2 in range(N2):
            xx = Y2[n2, :] - Y1[n1, :]
            # asymmetric chi2 -- take only variance of testing object!
            denominator = Y2_ivar[n2, :] #+ Y1_ivar[n1, :]
            ivar = Y2_ivar[n2, :] / (denominator + (denominator <= 0.)) #* Y1_ivar[n1, :]
            chi2[n1, n2] = np.sum(ivar * xx**2)
        if infinite_diagonal and n1 < N2:
            chi2[n1, n1] = np.Inf
    return chi2

def NN(index, chi2, labels):
    """
    Index is the second index into chi2!
    """
    N1 = labels.shape[0]
    foo, N2 = chi2.shape
    assert foo == N1
    assert index < N2
    return labels[np.argmin(chi2[:, index]), :], np.argmin(chi2[:, index])


# -------------------------------------------------------------------------------
# load spectra and labels (giant stars)
# -------------------------------------------------------------------------------

'''f = open('data/training_labels_apogee_tgas_giants.pickle', 'r')    
training_labels = pickle.load(f)
f.close()

f = open('data/apogee_spectra_norm_giants.pickle', 'r')    
spectra = pickle.load(f)
f.close()

#bins = np.linspace(0, 500, 21)
#plt.hist(training_labels['SNR'], bins)
#plt.axvline(100, color = 'r')
#plt.tick_params(axis=u'both', direction='in', which='both')
#plt.xlabel('S/N')
#plt.savefig('plots/NN_mag/SN_cut.png') 
#
## remove low S/N data + one super large outlier
##bad = np.logical_or(training_labels['SNR'] < 100., training_labels['SNR'] > 1000.)
#bad = training_labels['SNR'] > 1000.
#training_labels = training_labels[~bad]
#spectra = spectra[:, ~bad, :]                    

wl = spectra[:, 0, 0]
fluxes = spectra[:, :, 1].T
ivars = (1./(spectra[:, :, 2]**2)).T 
        
# -------------------------------------------------------------------------------
# load spectra and labels (validation set)
# -------------------------------------------------------------------------------                                  

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
latex["Q_K"] = r"$Q_{K}$"

plot_limits = {}
plot_limits['TEFF'] = (3000, 7000)
plot_limits['FE_H'] = (-2.5, 1)
plot_limits['LOGG'] = (0, 4.)
plot_limits['ALPHA_FE'] = (-.2, .6)
plot_limits['KMAG_ABS'] = (-1, -6)
plot_limits['Q_K'] = (0, 1.2)

labels = np.array(['TEFF', 'LOGG', 'FE_H', 'Q_K']) #, 'ALPHA_M', 'Q_MAG', 'N_FE', 'C_FE'])
Nlabels = len(labels)
latex_labels = [latex[l] for l in labels]
tr_label_input, tr_var_input = make_label_input(labels, all_training_labels)
#tr_label_validation, tr_var_validation = make_label_input(labels, training_labels_validation)
print(Nlabels, tr_label_input.shape, tr_var_input.shape, fluxes.shape, ivars.shape) 

# -------------------------------------------------------------------------------
# take random indices
# -------------------------------------------------------------------------------

np.random.seed(34)
indices = np.arange(all_fluxes.shape[0])
np.random.shuffle(indices)
ind_train = indices[:N]
ind_validation = indices[N:]

# -------------------------------------------------------------------------------
# input data
# -------------------------------------------------------------------------------

#X = fluxes
#X_ivar = ivars
#Y = tr_label_input
#N, D = fluxes.shape
#N, L = tr_label_input.shape
#
#X2 = fluxes_validation
#X2_ivar = ivars_validation
#N2, D = fluxes_validation.shape

X = all_fluxes[ind_train, :]
X_ivar = all_ivars[ind_train, :]
Y = tr_label_input[ind_train, :]
N, D = all_fluxes.shape
N, L = tr_label_input[ind_train, :].shape

X2 = all_fluxes[ind_validation, :]
X2_ivar = all_ivars[ind_validation, :]
N2, D = all_fluxes[ind_validation, :].shape

# -------------------------------------------------------------------------------
# validation with NN
# -------------------------------------------------------------------------------

chi2 = Chi2_Matrix(X, X_ivar, X2, X2_ivar)

plt.imshow(1./chi2, interpolation = None, cmap = 'viridis')
plt.colorbar()
plt.savefig('plots/NN_mag/chi2_asym_new_val.png')

new_labels = np.zeros((N2, L))

for i in range(N2):
    lab_i, index = NN(i, chi2, Y)
    new_labels[i, :] = lab_i


for i, l in enumerate(labels):

    orig = tr_label_input[ind_validation, i]
    gp_values = new_labels[:, i]
    scatter = np.round(np.std(orig - gp_values), 5)
    bias = np.round(np.mean(orig - gp_values), 5) 
    chi2_label = np.round(np.sum((orig - gp_values)**2 / tr_var_input[ind_validation, i]), 4)
    
    xx = [-10000, 10000]
    plt.figure(figsize=(6, 6))
    plt.scatter(orig, gp_values, color=colors[-2], label=' bias = {0} \n scatter = {1} \n $\chi^2$ = {2}'.format(bias, scatter, chi2_label), marker = 'o')
    plt.plot(xx, xx, color=colors[2], linestyle='--')
    plt.xlabel(r'reference labels {}'.format(latex[l]), size=14)
    plt.ylabel(r'inferred values {}'.format(latex[l]), size=14)
    plt.tick_params(axis=u'both', direction='in', which='both')
    plt.xlim(plot_limits[l])
    plt.ylim(plot_limits[l])
    plt.tight_layout()
    plt.legend(loc=2, fontsize=14, frameon=True)
    plt.savefig('plots/NN_mag/1to1_{0}_asym_new_val.png'.format(l))
    plt.close()
    
    if l == 'Q_K' or l == 'Q_L':
        orig = 5. * (np.log10(tr_label_input[ind_validation, i]))
        gp_values = 5. * (np.log10(new_labels[:, i]))
        scatter = np.round(np.std(orig - gp_values), 4)
        bias = np.round(np.mean(orig - gp_values), 4)    
        if l == 'Q_K':
            chi2_label = np.round(np.sum((orig - gp_values)**2 / all_training_labels['K_ERR'][ind_validation]), 4)
        elif l == 'Q_L':
            chi2_label = np.round(np.sum((orig - gp_values)**2 / all_training_labels['L_ERR'][ind_validation]), 4)        
        xx = [-10000, 10000]
        plt.figure(figsize=(6, 6))
        plt.scatter(orig, gp_values, color=colors[-2], label=' bias = {0} \n scatter = {1} \n $\chi^2$ = {2}'.format(bias, scatter, chi2_label), marker = 'o')
        plt.plot(xx, xx, color=colors[2], linestyle='--')
        if l == 'Q_K':
            plt.xlabel(r'reference labels $M_{K}$', size=12)
            plt.ylabel(r'inferred values $M_{K}$', size=12)
        elif l == 'Q_L':
            plt.xlabel(r'reference labels $M_{L}$', size=12)
            plt.ylabel(r'inferred values $M_{L}$', size=12)            
        plt.tick_params(axis=u'both', direction='in', which='both')
        plt.xlim(-5, 0)
        plt.ylim(-5, 0)
        plt.tight_layout()
        plt.legend(loc=2, fontsize=14, frameon=True)
        plt.title('#stars: {0}'.format(len(ind_validation)), fontsize=12)
        if l == 'Q_K':        
            plt.savefig('plots/NN_mag/1to1_M_K_asym_new_val.png')
        elif l == 'Q_L':        
            plt.savefig('plots/NN_mag/1to1_M_L_asym_new_val.png')
            plt.close()

## -------------------------------------------------------------------------------'''
## cross validation with NN
## -------------------------------------------------------------------------------
#
#chi2 = Chi2_Matrix(X, X_ivar, X, X_ivar, infinite_diagonal = True)
#
#plt.imshow(1./chi2, interpolation = None, cmap = 'viridis')
#plt.colorbar()
##plt.axhline(13, color = 'r')
##plt.axhline(34, color = 'r')
##plt.axhline(35, color = 'r')
#plt.savefig('plots/NN_mag/chi2_giants_asym.png')
#
#new_labels = np.zeros_like(Y)
#
#for i in range(N):
#    lab_i, index = NN(i, chi2, Y)
#    new_labels[i, :] = lab_i
#
#
#for i, l in enumerate(labels):
#
#    orig = Y[:, i]
#    gp_values = new_labels[:, i]
#    scatter = np.round(np.std(orig - gp_values), 5)
#    bias = np.round(np.mean(orig - gp_values), 5)    
#    
#    xx = [-10000, 10000]
#    plt.figure(figsize=(6, 6))
#    plt.scatter(orig, gp_values, color=colors[-2], label=' bias = {0} \n scatter = {1}'.format(bias, scatter), marker = 'o')
#    plt.plot(xx, xx, color=colors[2], linestyle='--')
#    plt.xlabel(r'reference labels {}'.format(latex[l]), size=14)
#    plt.ylabel(r'inferred values {}'.format(latex[l]), size=14)
#    plt.tick_params(axis=u'both', direction='in', which='both')
#    plt.xlim(plot_limits[l])
#    plt.ylim(plot_limits[l])
#    plt.tight_layout()
#    plt.legend(loc=2, fontsize=14, frameon=True)
#    plt.savefig('plots/NN_mag/1to1_{0}_asym.png'.format(l))
#    plt.close()

## -------------------------------------------------------------------------------
## find three spectra that seem to dominate...
## -------------------------------------------------------------------------------
#    
#chi2 = Chi2_Matrix(X, X_ivar, X, X_ivar, infinite_diagonal = False)
#
#sums = np.mean(chi2, axis = 0) 
#print np.argmin(sums)   
#print training_labels['SNR'][np.argmin(sums)] # minimum SNR!! 25
#print min(training_labels['SNR'])
#
## remove minimum
#bad = sums == np.min(sums)
#sums = sums[~bad]
#training_labels = training_labels[~bad]
#print len(sums)
#print np.argmin(sums)
#print training_labels['SNR'][np.argmin(sums)]  # 155
#
## remove minimum
#bad = sums == np.min(sums)
#sums = sums[~bad]
#training_labels = training_labels[~bad]
#print len(sums)
#print np.argmin(sums)
#print training_labels['SNR'][np.argmin(sums)]   
                    
## -------------------------------------------------------------------------------'''
## load spectra and labels (cluster stars)
## -------------------------------------------------------------------------------
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
#        
## exclude pleiades
#pleiades = table['CLUSTER'] == 'Pleiades'
#training_labels = training_labels[~pleiades]
#fluxes = fluxes[~pleiades]
#ivars = ivars[~pleiades]


'''# -------------------------------------------------------------------------------
# load spectra and labels 
# -------------------------------------------------------------------------------

# loading training labels
#f = open('/Users/eilers/Dropbox/cygnet/data/training_labels_apogee_tgas.pickle', 'r')
f = open('data/training_labels_apogee_tgas.pickle', 'r')
training_labels = pickle.load(f)
f.close()

# loading normalized spectra
#f = open('/Users/eilers/Dropbox/cygnet/data/apogee_spectra_norm.pickle', 'r')
f = open('data/apogee_spectra_norm.pickle', 'r')        
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

plot_limits = {}
plot_limits['TEFF'] = (3000, 7000)
plot_limits['FE_H'] = (-2.5, 1)
plot_limits['LOGG'] = (0, 4.)
plot_limits['ALPHA_FE'] = (-.2, .6)
plot_limits['KMAG_ABS'] = (-1, -6)
plot_limits['Q_MAG'] = (-3, 1)

labels = np.array(['TEFF', 'LOGG', 'FE_H', 'Q_MAG']) #, 'ALPHA_M', 'Q_MAG', 'N_FE', 'C_FE'])
Nlabels = len(labels)
latex_labels = [latex[l] for l in labels]
tr_label_input, tr_var_input = make_label_input(labels, training_labels)
print(Nlabels, tr_label_input.shape, tr_var_input.shape, fluxes.shape, ivars.shape) 

# -------------------------------------------------------------------------------
# input data
# -------------------------------------------------------------------------------

X = fluxes
X_ivar = ivars
Y = tr_label_input
#Y_var = tr_var_input
N, D = fluxes.shape

# -------------------------------------------------------------------------------
# cross validation with NN
# -------------------------------------------------------------------------------

chi2 = Chi2_Matrix(X, X_ivar, X, X_ivar, infinite_diagonal = True)

new_labels = np.zeros_like(Y)

for i in range(N):
    lab_i = NN(i, chi2, Y)
    new_labels[i, :] = lab_i


for i, l in enumerate(labels):

    orig = Y[:, i]
    gp_values = new_labels[:, i]
    scatter = np.round(np.std(orig - gp_values), 5)
    bias = np.round(np.mean(orig - gp_values), 5)    
    
    xx = [-10000, 10000]
    plt.figure(figsize=(6, 6))
    plt.scatter(orig, gp_values, color=colors[-2], label=' bias = {0} \n scatter = {1}'.format(bias, scatter), marker = 'o')
    plt.plot(xx, xx, color=colors[2], linestyle='--')
    plt.xlabel(r'reference labels {}'.format(latex[l]), size=14)
    plt.ylabel(r'inferred values {}'.format(latex[l]), size=14)
    plt.tick_params(axis=u'both', direction='in', which='both')
    plt.xlim(plot_limits[l])
    plt.ylim(plot_limits[l])
    plt.tight_layout()
    plt.legend(loc=2, fontsize=14, frameon=True)
    plt.savefig('plots/NN/1to1_{0}_RC.png'.format(l))
    plt.close()

# -------------------------------------------------------------------------------'''
