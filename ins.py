import pandas as pd
# from numрy import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

__G__ = 9.81

def accel_calibration_1():
    # z+, z-, x+, x-, y+, y-
    df = pd.read_csv('ins_data/second_accel.csv', sep=';', dtype=np.int32)

    a1 = df.ACC1
    a2 = df.ACC2
    a3 = df.ACC3

    a_slices = [slice(1300, 1600), slice(2200, 2500), slice(700, 1000), slice(1800, 2100), slice(2700, 3000), slice(0, 300)]

    raw_data = np.array([[x,y,z] for x,y,z in zip(a1, a2, a3)])

    averaged_a1 = np.array(list(map(lambda x: a1[x].mean(), a_slices)))
    averaged_a2 = np.array(list(map(lambda x: a2[x].mean(), a_slices)))
    averaged_a3 = np.array(list(map(lambda x: a3[x].mean(), a_slices)))

 

    a_plus = np.array(
        [
            averaged_a1[3:],
            averaged_a2[3:],
            averaged_a3[3:]
        ]
    )

    a_minus = np.array(
        [
            averaged_a1[:3],
            averaged_a2[:3],
            averaged_a3[:3]
        ]
    )

    S_a = (a_minus - a_plus) / 2 / __G__
    b_a = (a_plus + a_minus) / 2

    S_f = np.linalg.inv(S_a)
    b_f = -S_f @ b_a

    aggregated_data = np.array([S_f @ x + b_f[:, 0] for x in raw_data])
    
    
    plt.subplot(3, 2, 1)
    plt.title('Raw data')
    plt.plot(df.ACC1, color='r')
    plt.subplot(3, 2, 3)
    plt.plot(df.ACC2, color='g')
    plt.subplot(3, 2, 5)
    plt.plot(df.ACC3, color='b')
    plt.xlabel('Clocks')
    
    
    plt.subplot(3, 2, 2)
    plt.title('Calibrated data')
    plt.plot(aggregated_data[:, 0], color='r')
    plt.subplot(3, 2, 4)
    plt.plot(aggregated_data[:, 1], color='g')
    plt.subplot(3, 2, 6)
    plt.plot(aggregated_data[:, 2], color='b')
    plt.xlabel('Clocks')
    plt.show()

    return S_f, b_f[:, 0]


def gyro_calibration():
    # z+ z- y+ y- x+ x-
    df = pd.read_csv('./ins_data/first_gyro.csv', sep=';', dtype=np.int32)

    g1 = df.GYRO1
    g2 = df.GYRO2
    g3 = df.GYRO3
  
    w_slices = [slice(3900, 4100), slice(4630, 4770), slice(2540, 2680), slice(1890, 2010), slice(700, 830), slice(4390, 4480)] 

    b_w = np.array([g1[400:1000].mean(), g2[400:1000].mean(), g3[2800:3000].mean()])

    g1 -= b_w[0]
    g2 -= b_w[1]
    g3 -= b_w[2]

    raw_data = np.array([[x,y,z] for x,y,z in zip(g1, g2, g3)])

    averaged_g1 = np.array(list(map(lambda x: g1[x].mean(), w_slices)))
    averaged_g2 = np.array(list(map(lambda x: g2[x].mean(), w_slices)))
    averaged_g3 = np.array(list(map(lambda x: g3[x].mean(), w_slices)))

    phi_minus = np.array(
        [
            averaged_g1[::2],
            averaged_g2[::2],
            averaged_g3[::2]
        ]
    )

    phi_plus = np.array(
        [
            averaged_g1[1::2],
            averaged_g2[1::2],
            averaged_g3[1::2]
        ]
    )
    
    Sw = (phi_minus - phi_plus) / np.deg2rad(180)
    S_w = np.linalg.inv(Sw)

    
    aggregated_data = np.array([S_w @ x for x in raw_data])

    print(S_w)

    plt.subplot(3, 2, 1)
    plt.title('Raw data')
    plt.plot(g1, color='r')
    plt.subplot(3, 2, 3)
    plt.plot(g2, color='g')
    plt.subplot(3, 2, 5)
    plt.plot(g3, color='b')
    
    plt.subplot(3, 2, 2)
    plt.title('Calibrated data')
    plt.plot(aggregated_data[:, 0], color='r')
    plt.subplot(3, 2, 4)
    plt.plot(aggregated_data[:, 1], color='g')
    plt.subplot(3, 2, 6)
    plt.plot(aggregated_data[:, 2], color='b')
    plt.show()



def accel_calibration_2(S_f0, b_f0):
    def J(x):
        k = 0
        for i in range(18):
            M = x[:9].reshape(3, 3) @ A[:, i] + x[9:]
            k += (__G__**2 - M @ M)**2
        return k
    

    df = pd.read_csv('ins_data/third_accel_18.csv', sep=';', dtype=np.int32)

    a1 = df.ACC1
    a2 = df.ACC2
    a3 = df.ACC3

    a_slices = [
        slice(1, 300), slice(700, 980), slice(1090, 1465), 
        slice(1580, 1980), slice(2315, 3040), slice(3170, 3510), 
        slice(3610, 4000), slice(4100, 4640), slice(4715, 5145),
        slice(5230, 5680), slice(5775, 6170), slice(6260, 7530),
        slice(7700, 8330), slice(8450, 8860), slice(8945, 9440),
        slice(9530, 9940), slice(10040, 10480), slice(10580, 10970)
    ]

    raw_data = np.array([[x,y,z] for x,y,z in zip(a1, a2, a3)])

    A = np.array(list(map(lambda x: [a1[x].mean(), a2[x].mean(), a3[x].mean()], a_slices))).reshape(3, 18)
    
    x_0 = np.concatenate((S_f0.flatten(), b_f0.flatten()), axis=0)

    xopt = scipy.optimize.fmin(func=J, x0=x_0)

    S_f, b_f = xopt[:9].reshape(3, 3), xopt[9:]

    aggregated_data = np.array([S_f @ x + b_f for x in raw_data])

    plt.subplot(3, 2, 1)
    plt.title('Raw data')
    plt.plot(df.ACC1, color='r')
    plt.legend('acc1 raw')
    plt.subplot(3, 2, 3)
    plt.plot(df.ACC2, color='b')
    plt.legend('acc2 raw')
    plt.subplot(3, 2, 5)
    plt.plot(df.ACC3, color='b')
    plt.legend('acc3 raw')
    plt.xlabel('Clocks')
    
    
    plt.subplot(3, 2, 2)
    plt.title('Calibrated data')
    plt.plot(aggregated_data[:, 0], color='r')
    plt.subplot(3, 2, 4)
    plt.plot(aggregated_data[:, 1], color='g')
    plt.subplot(3, 2, 6)
    plt.plot(aggregated_data[:, 2], color='b')
    plt.xlabel('Clocks')
    plt.show()

if __name__ == "__main__":
    S, b = accel_calibration_1()
    # gyro_calibration()
    accel_calibration_2(S, b)