#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 20:10:52 2017

@author: eilers
"""

import numpy as np
from astropy.table import Table, hstack, join
import os.path
import subprocess
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
import pickle
import matplotlib.pyplot as plt
from astropy.io import fits

# -------------------------------------------------------------------------------
# download data
# -------------------------------------------------------------------------------

delete0 = 'find ./data/ -size 0c -delete'
substr = 'l31c'
subsubstr = '2'
fn = 'allStar-' + substr + '.' + subsubstr + '.fits'
url = 'https://data.sdss.org/sas/dr14/apogee/spectro/redux/r8/stars/' + substr \
    + '/' + substr + '.' + subsubstr + '/' + fn
destination = './data/' + fn
cmd = 'wget ' + url + ' -O ' + destination
if not os.path.isfile(destination):
    subprocess.call(cmd, shell = True) # warning: security
    subprocess.call(delete0, shell = True)
print("opening " + destination)
apogee_table = fits.open(destination)
apogee_data = apogee_table[1].data                         
apogee_data = apogee_data[apogee_data['DEC'] > -90.0]

# download TGAS data   
fn = 'stacked_tgas.fits'
url = 'http://s3.adrian.pw/' + fn
destination = './data/'+ fn
cmd = 'wget ' + url + ' -O ' + destination
if not os.path.isfile(destination):
    subprocess.call(cmd, shell = True)
    subprocess.call(delete0, shell = True)
print("opening " + destination)
tgas_table = fits.open(destination)
tgas_data = tgas_table[1].data
                         
# -------------------------------------------------------------------------------
# match TGAS and APOGEE
# -------------------------------------------------------------------------------
apogee_cat = SkyCoord(ra=apogee_data['RA']*u.degree, dec=apogee_data['DEC']*u.degree)
tgas_cat = SkyCoord(ra=tgas_data['RA']*u.degree, dec=tgas_data['DEC']*u.degree)

id_tgas, id_apogee, d2d, d3d = apogee_cat.search_around_sky(tgas_cat, 0.001*u.degree)

tgas_data = tgas_data[id_tgas]
apogee_data = apogee_data[id_apogee]    
print('matched entries: {}'.format(len(tgas_data)))

# -------------------------------------------------------------------------------
# hack! Cut down to only high quality data!
# -------------------------------------------------------------------------------

cut = tgas_data['parallax']/tgas_data['parallax_error'] > 10.
tgas_data = tgas_data[cut]              
apogee_data = apogee_data[cut] 
             
# -------------------------------------------------------------------------------
# get APOGEE spectra
# -------------------------------------------------------------------------------
'''delete0 = 'find ./data/spectra/ -size 0c -delete'
subprocess.call(delete0, shell = True)

for i, (fn2, loc, field) in enumerate(zip(apogee_data['FILE'], apogee_data['LOCATION_ID'], apogee_data['FIELD'])):
    
    print('Looking for object {}'.format(apogee_data['FILE']))
    
    fn = fn2.replace('apStar-r8', 'aspcapStar-r8-l31c.2')
    destination2 = './data/spectra/' + fn2.strip()
    destination = './data/spectra/' + fn.strip()
    print(loc, destination, os.path.isfile(destination))
    print(loc, destination2, os.path.isfile(destination2))
    
    if not (os.path.isfile(destination) or os.path.isfile(destination2)):
        if loc == 1:
            try:
                url4 = 'https://data.sdss.org/sas/dr14/apogee/spectro/redux/r8/stars/apo1m/' + field.strip() + '/' + fn2.strip()
                cmd = 'wget ' + url4 + ' -O ' + destination2
                #print cmd
                subprocess.call(cmd, shell = True)
                subprocess.call(delete0, shell = True)
            except:
                print(fn + " " + fn2 + " not found in any location")
        else: 
            urlbase = 'https://data.sdss.org/sas/dr14/apogee/spectro/redux/r8/stars/' + substr \
            + '/' + substr + '.' + subsubstr + '/' + str(loc).strip() + '/'
            url = urlbase + fn.strip()
            url2 = urlbase + fn2.strip()
            try:
                cmd = 'wget ' + url + ' -O ' + destination
                #print cmd
                subprocess.call(cmd, shell = True)
                subprocess.call(delete0, shell = True)
            except:
                try:
                    cmd = 'wget ' + url2 + ' -O ' + destination2
                    #print cmd
                    subprocess.call(cmd, shell = True)
                    subprocess.call(delete0, shell = True)
                except:
                    try:
                        url3 = 'https://data.sdss.org/sas/dr14/apogee/spectro/redux/r8/stars/apo25m/'  + str(loc).strip() + '/' + fn2
                        cmd = 'wget ' + url3 + ' -O ' + destination2
                        #print cmd
                        subprocess.call(cmd, shell = True)
                        subprocess.call(delete0, shell = True)
                    except:
                        print(fn + " " + fn2 + " not found in any location")

# remove missing files 
found = np.ones_like(np.arange(len(apogee_data)), dtype=bool)
destination = './data/spectra/'
for i in range(len(apogee_data['FILE'])):
    entry = destination + (apogee_data['FILE'][i]).strip()
    entry2 = entry.replace('apStar-r8', 'aspcapStar-r8-l31c.2').strip()
    print(entry, entry2)
    try:
        hdulist = fits.open(entry)
    except:
        try:
            hdulist = fits.open(entry2)
        except:
            print(entry + " " + entry2 + " not found or corrupted; deleting!")
            cmd = 'rm -vf ' + entry + ' ' + entry2
            subprocess.call(cmd, shell = True)
            print(i, apogee_data['FILE'][i], apogee_data['FIELD'][i], apogee_data['LOCATION_ID'][i])
            found[i] = False

tgas_data = tgas_data[found]                     
apogee_data = apogee_data[found]    

print('spectra found for: {}'.format(len(apogee_data)))
 
# -------------------------------------------------------------------------------
# normalize spectra: functions
# -------------------------------------------------------------------------------

def LoadAndNormalizeData(file_spectra, file_name, destination):
    
    all_flux = np.zeros((len(file_spectra), 8575))
    all_sigma = np.zeros((len(file_spectra), 8575))
    all_wave = np.zeros((len(file_spectra), 8575))
    
    i=0
    for entry in file_spectra:
        print i
        try:
            hdulist = fits.open(destination + entry.strip())
        except:
            entry = entry.replace('apStar-r8', 'aspcapStar-r8-l31c.2')
            hdulist = fits.open(destination + entry.strip())
        if len(hdulist[1].data) < 8575: 
            flux = hdulist[1].data[0]
            sigma = hdulist[2].data[0]
            print('something is weird...')
        else:
            flux = hdulist[1].data
            sigma = hdulist[2].data
        header = hdulist[1].header
        start_wl = header['CRVAL1']
        diff_wl = header['CDELT1']
        val = diff_wl * (len(flux)) + start_wl
        wl_full_log = np.arange(start_wl, val, diff_wl)
        wl_full = [10**aval for aval in wl_full_log]
        all_wave[i] = wl_full        
        all_flux[i] = flux
        all_sigma[i] = sigma
        i += 1
        
    data = np.array([all_wave, all_flux, all_sigma])
    data_norm, continuum = NormalizeData(data.T)
    
    f = open('data/' + file_name, 'w')
    pickle.dump(data_norm, f)
    f.close()
    
    return data_norm, continuum

def NormalizeData(dataall):
        
    Nlambda, Nstar, foo = dataall.shape
    
    pixlist = np.loadtxt('data/pixtest8_dr13.txt', usecols = (0,), unpack = 1)
    pixlist = map(int, pixlist)
    LARGE  = 3.0                          # magic LARGE sigma value
   
    continuum = np.zeros((Nlambda, Nstar))
    dataall_flat = np.ones((Nlambda, Nstar, 3))
    dataall_flat[:, :, 2] = LARGE
    for jj in range(Nstar):
        bad_a = np.logical_or(np.isnan(dataall[:, jj, 1]), np.isinf(dataall[:,jj, 1]))
        bad_b = np.logical_or(dataall[:, jj, 2] <= 0., np.isnan(dataall[:, jj, 2]))
        bad = np.logical_or(np.logical_or(bad_a, bad_b), np.isinf(dataall[:, jj, 2]))
        dataall[bad, jj, 1] = 1.
        dataall[bad, jj, 2] = LARGE
        var_array = LARGE**2 + np.zeros(len(dataall)) 
        var_array[pixlist] = 0.000
        
        take1 = np.logical_and(dataall[:,jj,0] > 15150, dataall[:,jj,0] < 15800)
        take2 = np.logical_and(dataall[:,jj,0] > 15890, dataall[:,jj,0] < 16430)
        take3 = np.logical_and(dataall[:,jj,0] > 16490, dataall[:,jj,0] < 16950)
        ivar = 1. / ((dataall[:, jj, 2] ** 2) + var_array) 
        fit1 = np.polynomial.chebyshev.Chebyshev.fit(x=dataall[take1,jj,0], y=dataall[take1,jj,1], w=ivar[take1], deg=2) # 2 or 3 is good for all, 2 only a few points better in temp 
        fit2 = np.polynomial.chebyshev.Chebyshev.fit(x=dataall[take2,jj,0], y=dataall[take2,jj,1], w=ivar[take2], deg=2)
        fit3 = np.polynomial.chebyshev.Chebyshev.fit(x=dataall[take3,jj,0], y=dataall[take3,jj,1], w=ivar[take3], deg=2)
        continuum[take1, jj] = fit1(dataall[take1, jj, 0])
        continuum[take2, jj] = fit2(dataall[take2, jj, 0])
        continuum[take3, jj] = fit3(dataall[take3, jj, 0])
        dataall_flat[:, jj, 0] = 1.0 * dataall[:, jj, 0]
        dataall_flat[take1, jj, 1] = dataall[take1,jj,1]/fit1(dataall[take1, 0, 0])
        dataall_flat[take2, jj, 1] = dataall[take2,jj,1]/fit2(dataall[take2, 0, 0]) 
        dataall_flat[take3, jj, 1] = dataall[take3,jj,1]/fit3(dataall[take3, 0, 0]) 
        dataall_flat[take1, jj, 2] = dataall[take1,jj,2]/fit1(dataall[take1, 0, 0]) 
        dataall_flat[take2, jj, 2] = dataall[take2,jj,2]/fit2(dataall[take2, 0, 0]) 
        dataall_flat[take3, jj, 2] = dataall[take3,jj,2]/fit3(dataall[take3, 0, 0]) 
        
        bad = dataall_flat[:, jj, 2] > 0.3 # MAGIC
        dataall_flat[bad, jj, 1] = 1.
        dataall_flat[bad, jj, 2] = LARGE
        
    for jj in range(Nstar):
        print "continuum_normalize_tcsh working on star", jj
        bad_a = np.logical_not(np.isfinite(dataall_flat[:, jj, 1]))
        bad_a = np.logical_or(bad_a, dataall_flat[:, jj, 2] <= 0.)
        bad_a = np.logical_or(bad_a, np.logical_not(np.isfinite(dataall_flat[:, jj, 2])))
        bad_a = np.logical_or(bad_a, dataall_flat[:, jj, 2] > 1.)                    # magic 1.
        # grow the mask
        bad = np.logical_or(bad_a, np.insert(bad_a, 0, False, 0)[0:-1])
        bad = np.logical_or(bad, np.insert(bad_a, len(bad_a), False)[1:])
        dataall_flat[bad, jj, 1] = 1.
        dataall_flat[bad, jj, 2] = LARGE
            
    return dataall_flat, continuum

# -------------------------------------------------------------------------------
# normalize spectra
# -------------------------------------------------------------------------------

file_name = 'apogee_spectra_norm_nocuts.pickle'

destination = './data/' + file_name
if not os.path.isfile(destination):
    data_norm, continuum = LoadAndNormalizeData(apogee_data['FILE'], file_name, destination = './data/spectra/')

# -------------------------------------------------------------------------------
# save files!
# -------------------------------------------------------------------------------
apogee_data = Table(apogee_data)
tgas_data = Table(tgas_data)
training_labels = hstack([apogee_data, tgas_data])

f = open('data/training_labels_apogee_tgas_nocuts.pickle', 'w')
pickle.dump(training_labels, f)
f.close()  

# -------------------------------------------------------------------------------'''

# add extinction from Lauren Anderson's paper!
hdulist = fits.open('data/photoParallaxAnderson17.fits')
xx = hdulist[1].data

# match in RA and DEC
lauren_cat = SkyCoord(ra=xx['ra']*u.degree, dec=xx['dec']*u.degree)
tgas_cat = SkyCoord(ra=tgas_data['RA']*u.degree, dec=tgas_data['DEC']*u.degree)
id_tgas, id_lauren, d2d, d3d = lauren_cat.search_around_sky(tgas_cat, 0.001*u.degree)

tgas_data = tgas_data[id_tgas]
xx = xx[id_lauren]
apogee_data = apogee_data[id_tgas]  
print('matched entries: {}'.format(len(tgas_data)))

extinction = xx['dust E(B-V)']
            
Q = 10**(0.2*apogee_data['K']) * tgas_data['parallax']/100.                    # assumes parallaxes is in mas
Q_err = tgas_data['parallax_error'] * 10**(0.2*apogee_data['K'])/100. 

plt.scatter(apogee_data['TEFF'], apogee_data['LOGG'], c = apogee_data['FE_H'], vmin = -2., vmax = .5)
plt.xlim(5700, 3700)
plt.xlabel(r'$T_{eff}$')
plt.ylabel(r'$\log g$')
plt.ylim(4, 0)
plt.colorbar()

plt.scatter(apogee_data['TEFF'], Q, c = apogee_data['FE_H'], vmin = -2., vmax = .5)
plt.xlim(5700, 3700)
plt.ylim(-0.1, 5)
plt.xlabel(r'$T_{eff}$')
plt.ylabel(r'$Q$')
plt.colorbar()
