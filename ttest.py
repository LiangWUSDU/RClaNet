#encoding=utf-8
import xlrd
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.sandbox.stats.multicomp import multipletests


bk=xlrd.open_workbook('D:\Desktop\Review\Review Statistical analysis/预测值.xlsx')
sh=bk.sheet_by_name('ADNI-MA')
CNN = sh.col_values(0)
VGG = sh.col_values(1)
ResNet = sh.col_values(2)
GoogleNet =  sh.col_values(3)
MHCNN =  sh.col_values(4)
mask =  sh.col_values(5)
CNN = CNN[1:]
VGG = VGG[1:]
ResNet = ResNet[1:]
GoogleNet = GoogleNet[1:]
MHCNN = MHCNN[1:]
mask = mask[1:]
p1 = ttest_ind([x for x in np.round(CNN)==mask], [x for x in np.round(MHCNN)==mask])
p2 = ttest_ind([x for x in np.round(VGG)==mask], [x for x in np.round(MHCNN)==mask])
p3 = ttest_ind([x for x in np.round(ResNet)==mask], [x for x in np.round(MHCNN)==mask])
p4 = ttest_ind([x for x in np.round(GoogleNet)==mask], [x for x in np.round(MHCNN)==mask])
p = [p1[1],p2[1],p3[1],p4[1]]
p_adjusted = multipletests(p,alpha=0.05,method='bonferroni')
print(p)
print(p_adjusted)