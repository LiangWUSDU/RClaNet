import numpy as np
from scipy.stats import ttest_ind

Y = [[0,1,0], [1,0,0], [0,1,1],[0,1,1],[0,1,0]]

pred = [[1,0,0], [1,0,0], [0,1,1],[1,0,0],[1,0,0]]

pred2 = [[0,1,0], [1,0,0], [0,1,1],[0,1,1],[0,1,0]]

print([x.all() for x in np.round(pred)==Y])
print([x.all() for x in np.round(pred2)==Y])
cc = ttest_ind([x.all() for x in np.round(pred)==Y], [x.all() for x in np.round(pred2)==Y])


print(cc)