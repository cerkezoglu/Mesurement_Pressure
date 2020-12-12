import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt, erfc, pi, exp, erf
from scipy.special import erfinv

#Load data

dataset1 = pd.read_csv('dataset1.csv', sep=',')
time1 = np.array(dataset1['Time'], dtype = 'float32')
voltage1 = np.array(dataset1['Voltage'], dtype='float32')

dataset2 = pd.read_csv('dataset2.csv', sep=',')
time2 = np.array(dataset2['Time'], dtype = 'float32')
voltage2 = np.array(dataset2['Voltage'], dtype='float32')

dataset3= pd.read_csv('dataset3.csv', sep=',')
time3= np.array(dataset3['Time'], dtype = 'float32')
voltage3= np.array(dataset3['Voltage'], dtype='float32')

dataset4= pd.read_csv('dataset4.csv', sep=',')
time4= np.array(dataset4['Time'], dtype = 'float32')
voltage4= np.array(dataset4['Voltage'], dtype='float32')




class DataElemination():
    def __init__(self, time, voltage):
        self.voltage = voltage
        self.time = time
        self.velocity = self.get_velocity(voltage)
        self.n =len(self.velocity)
        self.critic_disig = self.cal_critic_disig()
        self.std, self.xm = self.calculate_std_xm(self.velocity)
        self.di_sig = self.cal_di_sig(self.velocity)
        self.eliminated_time, self.eliminated_velocity = self.eliminate_data(self.time, self.velocity)
        self.eliminated_n = len(self.eliminated_velocity)
        self.eliminated_std, self.eliminated_xm = self.calculate_std_xm(self.eliminated_velocity)

    def my_err(self, eta):
        err = round(1 / 2 * erf(eta / sqrt(2)), 6)
        return err

    def errf_inv(self,z):
        eta = round(sqrt(2) * erfinv(2 * z), 3)
        return eta

    def cal_critic_disig(self):
        n = self.n
        p = 1 - 1 / (2 * n)
        p_i = p / 2
        eta_critic = self.errf_inv(p_i)
        return eta_critic

    def calculate_std_xm(self, x):
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

    def cal_di_sig(self,x):
        n = self.n
        std = self.std
        xm = self.xm

        di_sig = np.zeros(n)
        for i in range(n):
            di_sig[i] = (x[i]-xm)/std
        return np.round(di_sig, 6)

    def eliminate_data(self, time, x):
        n = self.n
        disig = self.di_sig
        critic_disig = self.critic_disig
        a = []
        b = []
        for i in range(n):
            if abs(disig[i]) < critic_disig:
                a.append(x[i])
                b.append(time[i])

        return b, a

    def plot(self, t, x):
        n = self.n
        plt.scatter(t,x)
        _, xmm = self.calculate_std_xm(x)
        plt.plot(t, x, '--', label='velocity')
        plt.plot(self.time, self.xm + self.critic_disig * self.std * np.ones(n), 'r')
        plt.plot(self.time, self.xm - self.critic_disig * self.std * np.ones(n), 'r', label='dmax lines')
        plt.plot(self.time, xmm * np.ones(n), 'black', label='average')
        plt.ylabel('Velocity(m/s)')
        plt.xlabel('Time(s)')
        plt.title('Pressure Measurement Lab')
        plt.legend()
        plt.show()


    def get_velocity(self, voltage):
        delp_mmh2o = 25.405 * voltage -0.75 #calibration equation
        delp_pa = delp_mmh2o * 9.81 # mmh2o to Pa
        T = 22 # Celcius
        p = 1 #atm
        T_K = p+273
        p_pa = p* 101325 #Pa
        R = 287 # J/kg-K
        rho = p_pa / (R * T_K)
        velocity = [sqrt(2*abs(delp)/rho) for delp in delp_pa] #v = sqrt((p0 - p1)/ rho)
        return velocity


eliminator1 = DataElemination(time1, voltage1)
a1 = eliminator1.eliminated_velocity
b1= eliminator1.eliminated_time
n1_after = eliminator1.eliminated_n
print(n1_after)
eliminator1.plot(eliminator1.time, eliminator1.velocity)
eliminator1.plot(b1,a1)

eliminator2 = DataElemination(time2, voltage2)
a2 = eliminator2.eliminated_velocity
b2 = eliminator2.eliminated_time
n2_after = eliminator2.eliminated_n
print(n2_after)
eliminator2.plot(eliminator2.time, eliminator2.velocity)
eliminator2.plot(b2,a2)

eliminator3 = DataElemination(time3, voltage3)
a3 = eliminator3.eliminated_velocity
b3 = eliminator3.eliminated_time
n3_after = eliminator3.eliminated_n
print(n3_after)
eliminator3.plot(eliminator3.time, eliminator3.velocity)
eliminator3.plot(b3,a3)



eliminator4 = DataElemination(time4, voltage4)
a4 = eliminator4.eliminated_velocity
b4= eliminator4.eliminated_time
n4_after = eliminator4.eliminated_n
print(n4_after)
eliminator4.plot(eliminator4.time, eliminator4.velocity)
eliminator4.plot(b4, a4)

print(eliminator1.eliminated_xm, eliminator2.eliminated_xm, eliminator3.eliminated_xm, eliminator4.eliminated_xm)



# plt.plot(b, a)
# plt.show()









# Table = np.array([np.array(time).T, np.array(voltage).T, np.array(delp_mmh2o).T, np.array(delp_pa).T, np.array(velocity).T, np.array(cal_di_sig(velocity)).T], dtype='float32')
#
# print(max(cal_di_sig(velocity)))
# print(calculate_std_xm(velocity))
# print(cal_critic_disig(n))
# pd.DataFrame(Table.T).to_csv('table_2.csv', header=['time', 'voltage', 'delp_mmh2o', 'delp_pa', 'velocity', 'di_sig'])