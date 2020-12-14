from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix,matthews_corrcoef, make_scorer
import os
import joblib
import argparse
import statistics
import numpy as np
import math
import time
import multiprocessing
from sklearn.model_selection import PredefinedSplit, cross_validate,cross_val_predict


#Parser for inputs -------------------------------------------------------------
parser=argparse.ArgumentParser()
parser.add_argument("-c") #C hyperparameter
parser.add_argument("-g") #Gamma hyperparameter
parser.add_argument("-d") #DSSP path
parser.add_argument("-p") #Profile path
parser.add_argument("-cv") #CV sets path

args=parser.parse_args()

c_hyper=float(args.c)
g_hyper=float(args.g)
dssp_path=args.d
profile_path=args.p
cv_path=args.cv

print("You have inserted this parameters")
print("-C hyperparameter:\t",c_hyper)
print("-Gamma hyperparameter:\t",g_hyper)
print("-Dssp_path:\t",dssp_path)
print("-Profile_path:\t",profile_path)
print("-Cross_val_path:\t",cv_path, "\n")


start_time = time.time()

#-------------------------------------------------------------------------------
def build_2_matrix(m3):
	m2_H=np.array([[0,0],[0,0]])
	m2_E=np.array([[0,0],[0,0]])
	m2_C=np.array([[0,0],[0,0]])

	#2 classes H
	m2_H[0][0]=m3[0][0]
	m2_H[0][1]=m3[0][1]+m3[0][2]
	m2_H[1][0]=m3[1][0]+m3[2][0]
	m2_H[1][1]=m3[1][1]+m3[1][2]+m3[2][1]+m3[2][2]

	#2 classes E
	m2_E[0][0]=m3[1][1]
	m2_E[0][1]=m3[1][0]+m3[1][2]
	m2_E[1][0]=m3[0][1]+m3[2][1]
	m2_E[1][1]=m3[0][0]+m3[0][2]+m3[2][0]+m3[2][2]

	#2 classes C
	m2_C[0][0]=m3[2][2]
	m2_C[0][1]=m3[2][0]+m3[2][1]
	m2_C[1][0]=m3[0][2]+m3[1][2]
	m2_C[1][1]=m3[0][0]+m3[0][1]+m3[1][0]+m3[1][1]

	return m2_H,m2_E,m2_C

def mcc(m):
  d=(m[0][0]+m[1][0])*(m[0][0]+m[0][1])*(m[1][1]+m[1][0])*(m[1][1]+m[0][1])
  return (m[0][0]*m[1][1]-m[0][1]*m[1][0])/math.sqrt(d)

def get_acc(cm):
    #print (sum(cm[0])+sum(cm[1])+sum(cm[2]))
	return float(cm[0][0]+cm[1][1]+cm[2][2])/(sum(cm[0])+sum(cm[1])+sum(cm[2]))

def sen(m):
	return m[0][0]/(sum(m[0]))

def ppv(m):
	return m[0][0]/(m[0][0]+m[1][0])

#Preparation of the input ------------------------------------------------------
def input_SVM(filename,profile_path,dssp_path,pssm_files):
    window=17
    def reading_files(dssp,profile): #reading dssp and probile files
        profile_list=[]
        for line in dssp:
            if line.startswith(">"):
                sec_str1=""
                sec_str=dssp.readline().rstrip()
                for el in sec_str:
                    if el=="-":
                        sec_str1=sec_str1+"C"
                    else:
                        sec_str1=sec_str1+el
        for line in profile:
            if "A" not in line:
                line=line.rstrip()
                line=line.split()[1:] 
                profile_list.append(line)
        profile_list=[[float(j) for j in i] for i in profile_list]
        return(sec_str1,profile_list)

    def class_assign(sec_str): #assigning the class 1,2 or 3
        ss_class=[]
        for ss in sec_str:
            if ss=="H":
                ss_class.append(1)
            if ss=="E":
                ss_class.append(2)
            if ss=="C":
                ss_class.append(3)
        return(ss_class)

    def add_window_spaces(profile_list,window): #Padding
        for i in range(window//2):
            profile_list=[[0.0 for i in range(20)]]+profile_list
            profile_list.append([0.0 for i in range(20)])
        return(profile_list)

    #Main loop
    y=[]
    X=[]
    if filename in pssm_files:
        profile=open(profile_path+filename+".pssm")
        dssp=open(dssp_path+filename+".dssp")
        sec_str,profile_list=reading_files(dssp,profile) #function recall
        ss_class=class_assign(sec_str) #function recall
        profile_list=add_window_spaces(profile_list,window) #function recall
        for el in range(len(ss_class)):
            window_list=profile_list[el:el+window]
            window_list=[el for li in window_list for el in li]
            y.append(ss_class[el])
            X.append(window_list)
        return(X,y)
    else:
        return([],[])


pssm_files=[]
for pssm_file in os.listdir(profile_path):
    pssm_files.append(pssm_file[:-5])

X_list=[]
y_list=[]
fold_set=[]
for filename in os.listdir(cv_path):
    fold=filename[3]
    cv_file=open(cv_path+filename)
    for line in cv_file:
        ID=line.rstrip()
        X_train_test,y_train_test=input_SVM(ID,profile_path,dssp_path,pssm_files)
        if len(X_train_test)!=0:
            fold_set+=[int(fold) for i in range(len(y_train_test))]
            X_list=X_list+X_train_test
            y_list=y_list+y_train_test
X=np.asarray(X_list)
y=np.asarray(y_list)

ps=PredefinedSplit(fold_set)

mySVC = SVC(C=c_hyper, kernel='rbf', gamma=g_hyper) #build the model SVC
y_pred = cross_val_predict(mySVC, X, y, cv=ps, n_jobs=2)
#print(y_pred)
conf_mat = confusion_matrix(y, y_pred,labels=[1,2,3])
print(conf_mat)
print("Accuracy:",get_acc(conf_mat))
m2_H,m2_E,m2_C=build_2_matrix(conf_mat)
print()
print("mat 2x2 for H")
print(m2_H)
print("mat 2x2 for E")
print(m2_E)
print("mat 2x2 for C")
print(m2_C)
print()
print("mcc C:",mcc(m2_C),"mcc H:",mcc(m2_H),"mcc E:",mcc(m2_E))
print("sen C:",sen(m2_C),"sen H:",sen(m2_H),"sen E:",sen(m2_E))
print("ppv C:",ppv(m2_C),"ppv H:",ppv(m2_H),"ppv E:",ppv(m2_E))
print()
print("--- %s seconds ---" % (time.time() - start_time))
