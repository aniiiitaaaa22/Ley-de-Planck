""" -Statistical Mechanics LAB
Q.Plot Planck’s law for Black Body radiation and compare it with Raleigh-Jeans Law at high
  temperature and low temperature.
 """
import numpy as np  
from scipy.constants import h, c, k, pi  
import matplotlib.pyplot as plt
newparams = {'axes.labelsize':15, 'axes.linewidth': 1.5, 'savefig.dpi':  
      1000,  
     'lines.linewidth': 2, 'figure.figsize': (18, 12),    
     'legend.frameon': True,  
      'legend.handlelength': 0.7}  

plt.rcParams.update(newparams)  

L0 = np.linspace(50,1500,250) #Wavelength in nano m
L = L0*1e-9 #wavelength in m


def planck_lamda(L,T):  
     a = (2 * pi * h * c**2) / L **5  
     b = (h * c)/ (L * k * T)  
     c1 = np.exp(b) - 1  
     d = a / c1  
     return d
plt.suptitle("""Ley de Planck y Ley de Raleigh-Jeans para radiación de cuerpo negro """,  
          size = 14,color='b',fontstyle ="italic")      
    
R_Ht = (8* pi *k *10000)/L**4 #Rayleigh's law at High temperature  
R_Lt = (8*pi*k*2000)/L**4 #Rayleigh's law at Low temperature

T0 = planck_lamda(L,0.1)
T500 = planck_lamda(L , 500)  
T1000 = planck_lamda(L , 1000)
T1500 = planck_lamda(L , 1500)
T2000 = planck_lamda(L , 2000)
T2500 = planck_lamda(L , 2500)
T3000 = planck_lamda(L , 3000)
T3500 = planck_lamda(L , 3500)
T4000 = planck_lamda(L , 4000)
T4500 = planck_lamda(L , 4500)
T5000 = planck_lamda(L , 5000)
T5500 = planck_lamda(L , 5500)
T6000 = planck_lamda(L , 6000)
T6500 = planck_lamda(L , 6500)
T7000 = planck_lamda(L , 7000)
T7500 = planck_lamda(L , 7500)
T8000 = planck_lamda(L , 8000)
T8500 = planck_lamda(L , 8500)
T9000 = planck_lamda(L , 9000)
T9500 = planck_lamda(L , 9500)
T10000 = planck_lamda(L , 10000)


""" ##### Sub plot creation ##### """
plt.subplot(2,2,(1,2))
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
plt.title("Ley de Planck")
plt.ylim(0,14000*10e10)
plt.xlim(50*1e-9,1500*1e-9)


plt.subplot(2,2,3)
plt.plot(L, (planck_lamda(L,2000)),label='Ley de Planck')
plt.plot(L, R_Lt , "--" , label="Ley de Rayleigh-Jeans")
plt.legend(title="Comparación para baja temperatura",loc="best" ,prop={'size':12})
plt.xlabel(r"$\lambda$ ")  
plt.ylabel(r"U($\lambda $,T )")
plt.title("T=2000 K (Para Baja temperatura)")
plt.ylim(0,14*10e8)
plt.xlim(0,1500*1e-9)

  
plt.subplot(2,2,4)
plt.plot(L, T10000 ,label='Ley de Planck')
plt.plot(L, R_Ht , "--" , label="Ley de Rayleigh-Jeans")
plt.legend(title="Comparación para alta temperatura",loc="best" ,prop={'size':12})
plt.xlabel(r"$\lambda$ ")  
plt.ylabel(r"U($\lambda $,T )")
plt.title("T=10000 K (Para Alta temperatura)")
plt.ylim(0,14*10e9)


"""###### Sub plot Adjusting ######"""
plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)


plt.show()            


