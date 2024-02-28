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

plt.suptitle("""Ley de Planck """,  
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

lambdas = np.array([lam1500, lam2000, lam2500, lam3000, lam3500, lam4000, lam4500, lam5000,
                    lam5500, lam6000, lam6500, lam7000, lam7500, lam8000, lam8500, lam9000,
                    lam9500, lam10000])
max_values = np.array([maximo(T1500), maximo(T2000), maximo(T2500), maximo(T3000), maximo(T3500),
                       maximo(T4000), maximo(T4500), maximo(T5000), maximo(T5500), maximo(T6000),
                       maximo(T6500), maximo(T7000), maximo(T7500), maximo(T8000), maximo(T8500),
                       maximo(T9000), maximo(T9500), maximo(T10000)])

# Ajuste lineal
coefficients = np.polyfit(lambdas, max_values, 1)
trendline = np.polyval(coefficients, lambdas)

# Gráfica de los puntos y la línea de tendencia
plt.plot(lambdas, max_values, marker='o', label='Datos')
plt.plot(lambdas, trendline, label='Línea de tendencia', linestyle='--', color='red')

plt.xlabel(r'$\lambda$')
plt.ylabel(r'$U(\lambda, T_{\text{max}})$')
plt.title('Ajuste Lineal a los Datos')
plt.legend()
plt.show()

# Pendiente de la línea de tendencia
slope = coefficients[0]
print(f"Pendiente de la línea de tendencia: {slope}")





plt.show()






