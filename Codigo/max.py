import numpy as np  
from scipy.constants import h, c, k, pi  
import matplotlib.pyplot as plt
from scipy.signal import argrelmax

newparams = {'axes.labelsize': 15, 'axes.linewidth': 1.5, 'savefig.dpi': 1000,  
             'lines.linewidth': 2, 'figure.figsize': (18, 12),    
             'legend.frameon': True,  
             'legend.handlelength': 0.7}  

plt.rcParams.update(newparams)  

L0 = np.linspace(50, 1500, 250)  # Wavelength in nano m
L = L0 * 1e-9  # wavelength in m

def planck_lamda(L, T):  
    a = (2 * pi * h * c**2) / L ** 5  
    b = (h * c) / (L * k * T)  
    c1 = np.exp(b) - 1  
    d = a / c1  
    return d

plt.suptitle("""Ley de Planck y Ley de Raleigh-Jeans para radiación de cuerpo negro """,  
             size=14, color='b', fontstyle="italic")      

R_Ht = (8 * pi * k * 10000) / L**4  # Rayleigh's law at High temperature  
R_Lt = (8 * pi * k * 2000) / L**4  # Rayleigh's law at Low temperature

T0 = planck_lamda(L, 0.1)
T500 = planck_lamda(L, 500)  
T1000 = planck_lamda(L, 1000)
T1500 = planck_lamda(L, 1500)
T2000 = planck_lamda(L, 2000)
T2500 = planck_lamda(L, 2500)
T3000 = planck_lamda(L, 3000)
T3500 = planck_lamda(L, 3500)
T4000 = planck_lamda(L, 4000)
T4500 = planck_lamda(L, 4500)
T5000 = planck_lamda(L, 5000)
T5500 = planck_lamda(L, 5500)
T6000 = planck_lamda(L, 6000)
T6500 = planck_lamda(L, 6500)
T7000 = planck_lamda(L, 7000)
T7500 = planck_lamda(L, 7500)
T8000 = planck_lamda(L, 8000)
T8500 = planck_lamda(L, 8500)
T9000 = planck_lamda(L, 9000)
T9500 = planck_lamda(L, 9500)
T10000 = planck_lamda(L, 10000)



""" ##### Sub plot creation ##### """
plt.subplot(2, 2, (1, 2))
plt.plot(L, T0, label='T=0.1 K') 
plt.plot(L, T500, label='T=500 K')  
plt.plot(L, T1000, label='T=1000 K')
plt.plot(L, T1500, label='T=1500 K')
plt.plot(L, T2000, label='T=2000 K')
plt.plot(L, T2500, label='T=2500 K')
plt.plot(L, T3000, label='T=3000 K')
plt.plot(L, T3500, label='T=3500 K')
plt.plot(L, T4000, label='T=4000 K')
plt.plot(L, T4500, label='T=4500 K')
plt.plot(L, T5000, label='T=5000 K')
plt.plot(L, T5500, label='T=5500 K')
plt.plot(L, T6000, label='T=6000 K')
plt.plot(L, T6500, label='T=6500 K')
plt.plot(L, T7000, label='T=7000 K')
plt.plot(L, T7500, label='T=7500 K')
plt.plot(L, T8000, label='T=8000 K')
plt.plot(L, T8500, label='T=8500 K')
plt.plot(L, T9000, label='T=9000 K')
plt.plot(L, T9500, label='T=9500 K')
plt.plot(L, T10000, label='T=10000 K')
plt.legend(loc="best", prop={'size': 7})
plt.xlabel(r"$\lambda$ ")  
plt.ylabel(r"U($\lambda $,T )")
plt.title("Planck Law of Radiation")
plt.ylim(0, 14000*10e10)
plt.xlim(50*1e-9, 1500*1e-9)



def maximo(valores):
    mayor = valores[0]
    for i in range(0,len(valores)):
        if valores[i] > mayor:
            mayor = valores[i]
    return mayor

def valor_max(T,S):
    lam= pow((2*c*k*T)/maximo(S),1/4)
    return lam

#lam500 = valor_max(500,T500)
#lam1000 = valor_max(1000,T1000)
lam1500 = valor_max(1500,T1500)
lam2000 = valor_max(2000,T2000)
lam2500 = valor_max(2500,T2500)
lam3000 = valor_max(3000,T3000)
lam3500 = valor_max(3500,T3500)
lam4000 = valor_max(4000,T4000)
lam4500 = valor_max(4500,T4500)
lam5000 = valor_max(5000,T5000)
lam5500 = valor_max(5500,T5500)
lam6000 = valor_max(6000,T6000)
lam6500 = valor_max(6500,T6500)
lam7000 = valor_max(7000,T7000)
lam7500 = valor_max(7500,T7500)
lam8000 = valor_max(8000,T8000)
lam8500 = valor_max(8500,T8500)
lam9000 = valor_max(9000,T9000)
lam9500 = valor_max(9500,T9500)
lam10000 = valor_max(10000,T10000)





plt.figure()
#plt.plot(lam500,maximo(T500), marker='o')
#plt.plot(lam1000,maximo(T1000), marker='o')
plt.plot(lam1500,maximo(T1500), marker='o')
plt.plot(lam2000,maximo(T2000), marker='o')
plt.plot(lam2500,maximo(T2500), marker='o')
plt.plot(lam3000,maximo(T3000), marker='o')
plt.plot(lam3500,maximo(T3500), marker='o')
plt.plot(lam4000,maximo(T4000), marker='o')
plt.plot(lam4500,maximo(T4500), marker='o')
plt.plot(lam5000,maximo(T5000), marker='o')
plt.plot(lam5500,maximo(T5500), marker='o')
plt.plot(lam6000,maximo(T6000), marker='o')
plt.plot(lam6500,maximo(T6500), marker='o')
plt.plot(lam7000, maximo(T7000),marker='o')
plt.plot(lam7500,maximo(T7500), marker='o')
plt.plot(lam8000,maximo(T8000), marker='o')
plt.plot(lam8500,maximo(T8500), marker='o')
plt.plot(lam9000,maximo(T9000), marker='o')
plt.plot(lam9500,maximo(T9500), marker='o')
plt.plot(lam10000,maximo(T10000), marker='o')

lambdas = np.array([lam1500, lam2000, lam2500, lam3000, lam3500, lam4000, lam4500, lam5000,
                    lam5500, lam6000, lam6500, lam7000, lam7500, lam8000, lam8500, lam9000,
                    lam9500, lam10000])
max_values = np.array([maximo(T1500), maximo(T2000), maximo(T2500), maximo(T3000), maximo(T3500),
                       maximo(T4000), maximo(T4500), maximo(T5000), maximo(T5500), maximo(T6000),
                       maximo(T6500), maximo(T7000), maximo(T7500), maximo(T8000), maximo(T8500),
                       maximo(T9000), maximo(T9500), maximo(T10000)])

plt.scatter(lambdas, max_values)
plt.plot(lambdas, max_values)

coefficients = np.polyfit(lambdas, max_values, 1)
trendline = np.polyval(coefficients, lambdas)
plt.plot(lambdas, trendline, label='Línea de tendencia', linestyle='--', color='red')

plt.title('Ajuste Lineal a los Datos')

plt.show()


slope = coefficients[0]
print(f"Pendiente de la línea de tendencia: {slope}")









