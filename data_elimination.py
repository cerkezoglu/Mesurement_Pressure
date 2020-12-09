import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt, erfc, pi, exp, erf
from scipy.special import erfinv

from scipy import stats

a = pd.read_csv('fixed_data.csv', sep=',')
time = a['time']
voltage = a['voltage']
n = len(voltage)
average = 1/n * np.sum(voltage)
summ = 0
for i in range(n):
    summ += (voltage[i]-average) * (voltage[i]-average)

sigma = sqrt(summ/n)
print('std: ', sigma, 'avg: ', average)

plt.scatter(time, voltage)
plt.plot(time, voltage, '--')
plt.show()

def my_err(eta):
    err = round(1/2*erf(eta/sqrt(2)), 6)
    return err

def errf_inv(z):
    eta = round(sqrt(2) * erfinv(2*z), 3)
    return eta


def cal_critic_disig(n):
    p = 1-1/(2*n)
    pi = p/2
    eta_critic = errf_inv(pi)
    return eta_critic

# p = (1-1/(2 * n))/2
# print(p)
# print(my_err(3.72))
# print(errf_inv(0.43448))

critic_disig = cal_critic_disig(len(voltage))
print(critic_disig)