#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 11:04:16 2017

@author: eilers
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from astropy.table import Table
import seaborn as sns

from functions_gplvm import make_label_input

colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple", "pale red", "orange", "red", "blue"]
colors = sns.xkcd_palette(colors)

# -------------------------------------------------------------------------------
# functions
# -------------------------------------------------------------------------------

def Chi2_Matrix(Y1, Y1_ivar, Y2, Y2_ivar, infinite_diagonal=False):
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
            denominator = Y2_ivar[n2, :] + Y1_ivar[n1, :]
            ivar = Y2_ivar[n2, :] * Y1_ivar[n1, :] / (denominator + (denominator <= 0.))
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

# -------------------------------------------------------------------------------
# input data
# -------------------------------------------------------------------------------

X = fluxes
X_ivar = ivars
Y = tr_label_input
Y_ivar = tr_var_input
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
    plt.savefig('plots/NN/1to1_{0}.png'.format(l))
    plt.close()

# -------------------------------------------------------------------------------'''
