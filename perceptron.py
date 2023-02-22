import numpy as np
import matplotlib.pyplot as plt
import random
import time

########## Data Generation ###########
LMS = True
N = 100
R = 30
if LMS:
    data = np.zeros(shape=(N,4))
else:
    data = np.zeros(shape=(N,3))
w0 = random.uniform(-5,5)
w1 = random.uniform(-5,5)
w2 = random.uniform(-0.1,0.1)
#print ([w1, w2, w0])

for i in range(0,N):
    rand1 = random.uniform(-R,R)
    rand2 = random.uniform(-R,R)
    const = 1
    if np.dot([w0, w1, w2], [rand1, rand2, const]) >= 0:
        label = 1
    else:
        label = -1
    if LMS:
        data[i] = [rand1, rand2, label, np.dot([w0, w1, w2], [rand1, rand2, const])]
    else:
        data[i] = [rand1, rand2, label]
    if label >= 0:
        plt.plot(rand1, rand2, 'ro')
    else:
        plt.plot(rand1, rand2, 'bo')

plt.pause(0.5)

slope = -(w2/w1)/(w2/w0)
y_int = -w2/w1
plt.xlim([-R,R])
plt.ylim([-R,R])
x = np.linspace(-R, R, N)
y = slope * x + y_int;
ideal_line = plt.plot(x, y, '-b')
plt.pause(0.5)

############# Perceptron SGD #############
errors = N
rate = 0.005
random.seed(time.perf_counter())
w0 = random.uniform(-5,5)
w1 = random.uniform(-5,5)
w2 = random.uniform(-0.1,0.1)
w_vec = [w1, w2, w0]
#print (w_vec)
slope_new = -(w0/w2)/(w0/w1)
y_int_new = -w0/w2
x_new = np.linspace(-R, R, N)
y_new = slope_new * x_new + y_int_new;
line = plt.plot(x_new, y_new, '-g')
plt.pause(0.5)
while errors > 0:
    ref = line.pop(0)
    ref.remove()
    errors = 0
    for i in range (0,N):
        x = data[i,:2]
        x = np.append(x, [1], axis=None)
        pred = np.dot(w_vec, x)
        if pred >= 0:
            pred = 1
        else:
            pred = -1
        if pred != data[i,2]:
            errors = errors + 1
            if LMS:
                error = data[i,3] - pred;
                w_vec = np.add(w_vec, 2*rate*x*error)
            else:
                w_vec = np.add(w_vec, rate*x*data[i,2])
    slope_new = -(w_vec[2]/w_vec[1])/(w_vec[2]/w_vec[0])
    y_int_new = -w_vec[2]/w_vec[1]
    x_new = np.linspace(-R, R, N)
    y_new = slope_new * x_new + y_int_new;
    line = plt.plot(x_new, y_new, '-g')
    plt.pause(0.05)

print("done")
plt.close()
