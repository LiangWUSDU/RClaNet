#encoding=utf-8
import xlrd
import numpy as np
from keras.utils import to_categorical
healthy_diag = ['Cognitively normal',
				'No dementia',
				'uncertain dementia',
				'Unc: ques. Impairment']  # list of healthy descriptors

alz_diag = ['AD Dementia', 'AD dem Language dysf prior',
			'AD dem Language dysf with', 'AD dem w/oth (list B) not contrib',
			'AD dem w/oth unusual features', 'AD dem visuospatial, after',
			'AD dem w/oth unusual features/demt on',
			'AD dem w/Frontal lobe/demt at onset',
			'AD dem/FLD prior to AD dem',
			'AD dem w/oth unusual feat/subs demt',
			'AD dem w/depresss- not contribut',
			'AD dem distrubed social- with',
			'AD dem distrubed social- prior',
			'AD dem w/CVD not contrib',
			'AD dem w/CVD contribut',
			'AD dem visuospatial- with',
			'AD dem visuospatial- prior',
			'AD dem Language dysf after',
			'AD dem w/oth (list B) contribut',
			'AD dem distrubed social- after',
			'AD dem w/depresss- contribut',
			'AD dem w/depresss  not contribut',
			'AD dem w/depresss  contribut',
			'AD dem w/PDI after AD dem contribut',
			'AD dem w/PDI after AD dem not contrib',
			'AD dem w/depresss, not contribut',
			'DAT w/depresss not contribut',
			'DAT'  # DAT = dementia alzheimer's type
			]  # list of AD descriptors
def BN(x):
	max = np.max(x)
	min = np.min(x)
	x = (x - min) / (max - min)
	return x
def open_excel_1(fileName,length):
    y_status = {}
    bk=xlrd.open_workbook(fileName)
    sh=bk.sheet_by_name("OASIS_3")
    age = sh.col_values(3)
    mmse = sh.col_values(8)
    CDR = sh.col_values(10)
    status = sh.col_values(12)
    apoe = sh.col_values(22)
    age = age[1:]
    mmse = mmse[1:]
    CDR = CDR[1:]
    status = status[1:]
    apoe = apoe[1:]
    diagnosis_dict = {}  # create dictionary with key for each diagnostic option
    for x in status:
        diagnosis_dict[x] = ''
    for x in healthy_diag:
        y_status[x] = 0  # assign 0 to keys in diagnosis_dict belonging to healthy descriptors list
    for x in alz_diag:
        y_status[x] = 1  # assign alzheimer's descriptors diagnosis 1
    yy_status = []  # use the diagnostic dictionary to create list of all labels for elements in df_clin
    for i in status:
        yy_status.append(y_status[i])
    yy_status = np.array(yy_status)
    mmse = np.array(mmse)
    CDR = np.array(CDR)
    age =  np.array(age)
    apoe =  np.array(apoe)
    yy_status = np.reshape(yy_status,(length,1))
    return mmse,CDR,age,apoe,yy_status
def open_excel(fileName,length):
    y_CDR = np.zeros((length,))
    bk=xlrd.open_workbook(fileName)
    sh=bk.sheet_by_name("OASIS_3")
    age = sh.col_values(3)
    mmse = sh.col_values(8)
    CDR = sh.col_values(10)
    apoe = sh.col_values(22)
    age = age[1:]
    mmse = mmse[1:]
    CDR = CDR[1:]
    apoe = apoe[1:]
    mmse = np.array(mmse)
    CDR = np.array(CDR)
    age =  np.array(age)
    apoe =  np.array(apoe)
    age = BN(age)
    y_CDR[CDR==0]=0
    y_CDR[CDR==0.5]=1
    y_CDR[CDR==1]=1
    y_CDR[CDR==2]=1
    y_CDR = np.array(y_CDR)
    y_CDR = to_categorical(y_CDR)
    return mmse,age,apoe,y_CDR
def open_excel_AD(fileName,length):
    y_CDR = np.zeros((length))
    bk=xlrd.open_workbook(fileName)
    sh=bk.sheet_by_name("OASIS_3")
    age = sh.col_values(3)
    mmse = sh.col_values(8)
    CDR = sh.col_values(10)
    apoe = sh.col_values(22)
    age = age[1:]
    mmse = mmse[1:]
    CDR = CDR[1:]
    apoe = apoe[1:]
    mmse = np.array(mmse)
    CDR = np.array(CDR)
    age =  np.array(age)
    apoe =  np.array(apoe)
    age = BN(age)
    y_CDR[CDR==0]=0
    y_CDR[CDR==0.5]=1
    y_CDR[CDR==1]=2
    y_CDR[CDR==2]=2
    y_CDR = np.array(y_CDR)
    return mmse,age,apoe,y_CDR