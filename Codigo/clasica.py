import numpy as np  
from scipy.constants import c, k, pi  
import matplotlib.pyplot as plt
newparams = {'axes.labelsize':15, 'axes.linewidth': 1.5, 'savefig.dpi':  
      1000,  
     'lines.linewidth': 2, 'figure.figsize': (18, 12),    
     'legend.frameon': True,  
      'legend.handlelength': 0.7}  

plt.rcParams.update(newparams)  

L0 = np.linspace(50,1500,250) #Wavelength in nano m
L = L0*1e-9 #wavelength in m


def raleigh_lamda(L,T):  
     a = (2 * pi * c * k * T) / L**4   
     return a
plt.suptitle("""Ley de Planck y Ley de Raleigh-Jeans para radiación de cuerpo negro """,  
          size = 14,color='b',fontstyle ="italic")

T0 = raleigh_lamda(L,0.1)
T500 = raleigh_lamda(L , 500)  
T1000 = raleigh_lamda(L , 1000)
T1500 = raleigh_lamda(L , 1500)
T2000 = raleigh_lamda(L , 2000)
T2500 = raleigh_lamda(L , 2500)
T3000 = raleigh_lamda(L , 3000)
T3500 = raleigh_lamda(L , 3500)
T4000 = raleigh_lamda(L , 4000)
T4500 = raleigh_lamda(L , 4500)
T5000 = raleigh_lamda(L , 5000)
T5500 = raleigh_lamda(L , 5500)
T6000 = raleigh_lamda(L , 6000)
T6500 = raleigh_lamda(L , 6500)
T7000 = raleigh_lamda(L , 7000)
T7500 = raleigh_lamda(L , 7500)
T8000 = raleigh_lamda(L , 8000)
T8500 = raleigh_lamda(L , 8500)
T9000 = raleigh_lamda(L , 9000)
T9500 = raleigh_lamda(L , 9500)
T10000 = raleigh_lamda(L , 10000)


""" ##### Sub plot creation ##### """
plt.subplot()
plt.plot(L, T0,label='T=0.1 K') 
plt.plot(L, T500,label='T=500 K')  
plt.plot(L, T1000 ,label='T=1000 K')
plt.plot(L, T1500 ,label='T=1500 K')
plt.plot(L, T2000 ,label='T=2000 K')
plt.plot(L, T2500 ,label='T=2500 K')
plt.plot(L, T3000 ,label='T=3000 K')
plt.plot(L, T3500 ,label='T=3500 K')
plt.plot(L, T4000 ,label='T=4000 K')
plt.plot(L, T4500 ,label='T=4500 K')
plt.plot(L, T5000 ,label='T=5000 K')
plt.plot(L, T5500 ,label='T=5500 K')
plt.plot(L, T6000 ,label='T=6000 K')
plt.plot(L, T6500 ,label='T=6500 K')
plt.plot(L, T7000 ,label='T=7000 K')
plt.plot(L, T7500 ,label='T=7500 K')
plt.plot(L, T8000 ,label='T=8000 K')
plt.plot(L, T8500 ,label='T=8500 K')
plt.plot(L, T9000 ,label='T=9000 K')
plt.plot(L, T9500 ,label='T=9500 K')
plt.plot(L, T10000 ,label='T=10000 K')
plt.legend(loc="best" ,prop={'size':7})
plt.xlabel(r"$\lambda$ ")  
plt.ylabel(r"U($\lambda $,T )")
plt.title("Ley de Rayleigh-Jeans")
plt.ylim(0,20000*10e10)
plt.xlim(50*1e-9,1500*1e-9) 

plt.show() 