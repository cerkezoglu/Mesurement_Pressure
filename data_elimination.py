import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt, erfc, pi, exp, erf
from scipy.special import erfinv

a = pd.read_csv('fixed_data.csv', sep=',')
time = a['time']
voltage = a['voltage']

def my_err(eta):
    err = round(1 / 2 * erf(eta / sqrt(2)), 6)
    return err

def errf_inv(z):
    eta = round(sqrt(2) * erfinv(2 * z), 3)
    return eta

def cal_critic_disig(n):
    p = 1 - 1 / (2 * n)
    pi = p / 2
    eta_critic = errf_inv(pi)
    return eta_critic

def calculate_std_xm(x):
    n = len(x)
    a = 0
    xm = 1 / n * np.sum(x)
    for i in range(n):
        a += (x[i] - xm) * (x[i] - xm)
    if n <= 20:
        std = sqrt(1 / (n - 1) * a)
    else:
        std = sqrt(1 / n * a)
    return std, xm

def cal_di_sig(x):
    n = len(x)
    std, xm = calculate_std_xm(x)
    di_sig = np.zeros(n)
    for i in range(n):
        di_sig[i] = (x[i]-xm)/std
    return np.round(di_sig, 6)

delp_mmh2o = 25.407 * voltage -1.245 #calibration equation
delp_pa = delp_mmh2o * 9.81 # mmh2o to Pa
rho = 1.204 #Air @ 20 C

velocity = [sqrt(2*abs(delp)/rho) for delp in delp_pa] #v = sqrt((p0 - p1)/ rho)

def eliminate_data(x):
    n = len(x)
    sigma, xm = calculate_std_xm(x)
    disig = cal_di_sig(x)
    critic_disig = cal_critic_disig(n)

    for i in range(n):
        if abs(disig[i]) >= critic_disig:
            x.pop(i)

    return x
n = len(velocity)
sigma, average = calculate_std_xm(velocity)
critic_disig = cal_critic_disig(n)
plt.scatter(time, velocity)
plt.plot(time, velocity, '--', label = 'velocity')
plt.plot(time, average+critic_disig*sigma*np.ones(n), 'r')
plt.plot(time, average-critic_disig*sigma*np.ones(n), 'r', label = 'dmax lines')
plt.plot(time, average*np.ones(n), 'black', label = 'average')
plt.ylabel('Velocity(m/s)')
plt.xlabel('Time(s)')
plt.title('Pressure Measurement Lab')
plt.legend()
plt.show()



Table = np.array([np.array(time).T, np.array(voltage).T, np.array(delp_mmh2o).T, np.array(delp_pa).T, np.array(velocity).T, np.array(cal_di_sig(velocity)).T], dtype='float32')

pd.DataFrame(Table.T).to_csv('table.csv', header=['time', 'voltage', 'delp_mmh2o', 'delp_pa', 'velocity', 'di_sig'])