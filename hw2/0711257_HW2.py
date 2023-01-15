import numpy as np
# from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
x_train, x_test, y_train, y_test = np.load('classification_data.npy', allow_pickle=True)
# y_train=np.asarray(y_train).reshape(-1,1)
# y_test=np.asarray(y_test).reshape(-1,1)
# print(x_train.shape)
# print(x_test.shape)


# 1. Compute the mean vectors mi, (i=1,2) of each 2 classes
x1 = x_train[y_train==0]
m1 = np.mean(x1, axis = 0)
x2 = x_train[y_train==1]
m2 = np.mean(x2, axis = 0)
print(f"mean vector of class 1: {m1}", f"mean vector of class 2: {m2}\n\n")

# 2. Compute the Within-class scatter matrix SW
s1 = (x1 - m1).T@(x1 - m1)
s2 = (x2 - m2).T@(x2 - m2)
sw = s1 + s2
print(f"Within-class scatter matrix SW: \n{sw}\n\n")

# 3. Compute the Between-class scatter matrix SB
d = (m2 - m1).reshape(2,1)
sb = d@d.T
print(f"Between-class scatter matrix SB: \n{sb}\n\n")

# 4. Compute the Fisher’s linear discriminant
w = np.linalg.inv(sw)@d
w  /= np.linalg.norm(w)
print(f" Fisher’s linear discriminant: \n{w}\n\n")
 
# 5. Project the test data by linear discriminant and get the class prediction by K nearest-neighbor rule. Please report the accuracy score with K values from 1 to 5
# you can use accuracy_score function from sklearn.metric.accuracy_score
pro_train_t = (x_train@w)  #projection parameter t
pro_test_t = (x_test@w)

for k in range(1,6):
    y_pred = np.zeros(x_test.shape[0])
    for  i, x in enumerate(pro_test_t):
        dis = []
        knn = []
        for j, y in enumerate(pro_train_t):
            # print(j,abs(y - xx))
            dis.append([j,abs(y - x)[0]])
            # dis.append([j,np.sum((y - x)**2)])
        dis.sort(key = lambda a:a[1])   # sort by distance
        for kk in range(k):
            knn.append(y_train[dis[kk][0]])
        y_pred[i] = np.argmax(np.bincount(knn))   # mode
        # y_pred[i] = stats.mode(knn, keepdims=1)[0][0]

    print(f"K={k} accuracy rate: {accuracy_score(y_test, y_pred)}")

# 6. Plot the 1) best projection line on the training data and show the slope and intercept on the title (you can choose any value of intercept for better visualization) 2) colorize the data with each class 3) project all data points on your projection line. Your result should look like this image
c = -10
p = "mediumslateblue"
o = "orangered"
pro_train_t = (x_train@w) - c * w[1]  
pro_test_t = (x_test@w) - c * w[1]
pro_x_train = pro_train_t@w.T
pro_x_train[:,1] += c
pro_x_test = pro_test_t@w.T
pro_x_test[:,1] += c
upper_bound = np.max(pro_x_train[:, 0]) + 0.5
lower_bound = np.min(pro_x_train[:, 0]) - 0.5
x = [lower_bound, upper_bound]
slope = w[1] / w[0]
y = [slope*x[0] + c, slope*x[1] + c]
# project line
plt.plot(x, y, lw=1, c='k')

for i in range(x_train.shape[0]):
    if y_train[i]:
        plt.plot(
            [x_train[i, 0], pro_x_train[i, 0]],
            [x_train[i, 1], pro_x_train[i, 1]], lw = 0.5, alpha = 0.1, c = p)
    else:
        plt.plot(
            [x_train[i, 0], pro_x_train[i, 0]],
            [x_train[i, 1], pro_x_train[i, 1]], lw = 0.5, alpha = 0.1,c = o)

# data point
plt.scatter(x1[:, 0], x1[:, 1], s = 5, c = o, label = 'class 1')
plt.scatter(x2[:, 0], x2[:, 1], s = 5, c = p, label = 'class 2')

# projected data point
proj_x1_train = pro_x_train[y_train == 0]
proj_x2_train = pro_x_train[y_train == 1]
plt.scatter(proj_x1_train[:, 0], proj_x1_train[:, 1], s = 5, c = o)
plt.scatter(proj_x2_train[:, 0], proj_x2_train[:, 1], s = 5, c = p)

if c >= 0:
    title = f'Projection line: y={slope[0]:.8f}x+{c}'
else: 
    title = f'Projection line: y={slope[0]:.8f}x{c}'
plt.title(title)
plt.legend(loc ='lower right')
plt.gca().set_aspect('equal', adjustable = 'box')
# plt.savefig('plot.png', dpi=300, transparent=True)
plt.show()
# print(slope)