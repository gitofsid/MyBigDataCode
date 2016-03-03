"""
/* ******************************************************************************************
 * Name - Siddharth Baronia 
 *
 * Problems - 1. Load npz data
 *            2. Train linear kernel SVM with different penalty value
 *            3. Test using validation data
 *            4. Plot accuracy vs penalty for each penalty
 *            5. Repeat all above with feature scaling
 * ********************************************************************************************/ 
"""

import sys
from pyspark import SparkConf, SparkContext
import numpy as np
from sklearn import svm, preprocessing
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


penalty_arr = []

'''sbaronia - draw graph'''
def draw_graphs(penalty_arr,accuracy_arr):
    f,ax = plt.subplots()
    colors = ['r','b']
    for i in range(len(accuracy_arr)):
    	ax.semilogx(penalty_arr,accuracy_arr[i], marker='o', linestyle='--', color=colors[i], label='accuracy'+str(i),linewidth=2)
    plt.xlabel('Penalty(C)',fontweight='bold')
    plt.ylabel('Accuracy',fontweight='bold')
    plt.legend(loc='best')
    title = "Penalty vs Accuracy"
    plt.title(title,fontweight='bold')
    plt.show()
    return

'''sbaronia - train and test using LinearSVC and find accuracy error'''
def cal_linearSVC(train_data, train_label, val_data, val_label):
	accuracy_arr = []
	
	for c in penalty_arr:
		classifier = svm.LinearSVC(C=c)
		classifier.fit(train_data, train_label)
		prd = classifier.predict(val_data)
		acc_score = accuracy_score(val_label,prd)
		print acc_score
		accuracy_arr.append(acc_score)

	return accuracy_arr

def main():

	conf = SparkConf().setAppName('linear kernel SVM')
	sc = SparkContext(conf=conf)
	assert sc.version >= '1.5.1'

	input_path = sys.argv[1]

	global penalty_arr

	data_load = np.load(input_path)

	penalty_arr = [10 ** i for i in range(-12,6)]

	'''sbaronia - find accuracy errors without and with feature scaling'''
	print ("No feature scaling \n")
	accuracy_arr = cal_linearSVC(data_load['data_training'],data_load['label_training'],data_load['data_val'],data_load['label_val'])

	scaler = preprocessing.StandardScaler().fit(data_load['data_training'])
	sc_transform = scaler.transform(data_load['data_training'])

	scaler_test = preprocessing.StandardScaler().fit(data_load['data_val'])
	sc_transform_test = scaler.transform(data_load['data_val'])
	
	print ("With feature scaling \n")
	accuracy_arr_norm = cal_linearSVC(sc_transform, data_load['label_training'], sc_transform_test, data_load['label_val'])

	draw_graphs(penalty_arr,[accuracy_arr,accuracy_arr_norm])
  
if __name__ == "__main__":
    main()

'''
 Result-

No feature scaling

0.4906
0.4906
0.5038
0.5094
0.5712
0.6108
0.6424
0.661
0.1768
0.5278
0.389
0.5506
0.57
0.4196
0.5646
0.2798
0.5362
0.438

With feature scaling

0.6448
0.6448
0.6448
0.645
0.645
0.646
0.649
0.6608
0.682
0.707
0.7152
0.7172
0.7164
0.6976
0.5754
0.6088
0.5872
0.5866


'''
