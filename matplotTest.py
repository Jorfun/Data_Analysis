import matplotlib.pyplot as plt
import numpy as np

# 最简单的用法
#
# plt.plot([1,2,3,4])
# plt.ylabel('some numbers')
# plt.show()



# For every x, y pair of arguments
#
# plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
# plt.show()



# An optional third argument  &  The axis() [xmin, xmax, ymin, ymax] 
#
# plt.plot([1,2,3,4], [1,4,9,16], 'ro')
# plt.axis([0, 6, 0, 20])
# plt.show()



# plotting several lines with different format styles in one command using arrays
#
# # evenly sampled time at 200ms intervals
# t = np.arange(0., 5., 0.2)

# # red dashes, blue squares and green triangles
# plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
# plt.show()



# adjust linewidth
#
# x = [1, 2, 3, 4]; y = [1, 4, 9, 16]
# plt.plot(x, y, linewidth=10.0)
# plt.show()



# Use the setter methods of a Line2D instance. plot returns a list of Line2D objects
#
# x = [1, 2, 3, 4]; y = [1, 4, 9, 16]
# line, = plt.plot(x, y, '-')
# line.set_antialiased(False) # turn off antialising
# plt.show()



# set multiple properties on a list of lines
#
# x1 = [1, 2, 3, 4]; y1 = [1, 4, 9, 16]; x2 = [3, 1, 5, 6]; y2 = [16, 8, 4, 1];
# lines = plt.plot(x1, y1, x2, y2)
# # use keyword args
# plt.setp(lines, color='r', linewidth=2.0)
# # or MATLAB style string value pairs
# plt.setp(lines, 'color', 'r', 'linewidth', 2.0)
# plt.show()



def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

plt.figure(1)
plt.subplot(211)
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

plt.subplot(212)
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
plt.show()