"""
/* ******************************************************************************************
 * Name - Siddharth Baronia 
 *
 * Problems - 1. Load npz data
 *            2. Train Kernelized SVM with different penalty value and gamma value
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

'''sbaronia - draw graphs'''
def draw_graphs(penalty_arr,accuracy_arr):
    f,ax = plt.subplots()
    colors = ['r','c','k','g','y','m','b','lime']
    for i in range(len(accuracy_arr)):
    	ax.semilogx(penalty_arr,accuracy_arr[i], marker='o', linestyle='--', color=colors[i], label='gamma'+str(i+1),linewidth=2)
    plt.xlabel('Penalty',fontweight='bold')
    plt.ylabel('Accuracy',fontweight='bold')
    plt.legend(loc='best')
    title = "Penalty(C) vs Accuracy"
    plt.title(title,fontweight='bold')
    plt.show()
    return

def main():

	conf = SparkConf().setAppName('Kernelized SVM')
	sc = SparkContext(conf=conf)
	assert sc.version >= '1.5.1'
	input_path = sys.argv[1]

	data_load = np.load(input_path)

	gammas_arr = [10 ** i for i in range(-8,0)]
	penalty_arr = [10 ** i for i in range(-12,6)]
	accuracy_arr = []


	'''sbaronia - for every value of gamma and penalty train and test'''
	for g in gammas_arr:
		temp = []
		for c in penalty_arr:
			classifier = svm.SVC(kernel='rbf', gamma=g, C=c)
			classifier.fit(data_load['data_training'][:2000], data_load['label_training'][:2000])
			prd = classifier.predict(data_load['data_val'])
			acc_score = accuracy_score(data_load['label_val'],prd)
			print "For gamma " + str(g) + " penalty(C) " + str(c) + " accuracy " + str(acc_score)
			temp.append(acc_score)
		print "\n"
		accuracy_arr.append(temp)

	draw_graphs(penalty_arr,accuracy_arr)
  
if __name__ == "__main__":
    main()


'''
For gamma 1e-08 penalty(C) 1e-12 accuracy 0.4906
For gamma 1e-08 penalty(C) 1e-11 accuracy 0.4906
For gamma 1e-08 penalty(C) 1e-10 accuracy 0.4906
For gamma 1e-08 penalty(C) 1e-09 accuracy 0.4906
For gamma 1e-08 penalty(C) 1e-08 accuracy 0.4906
For gamma 1e-08 penalty(C) 1e-07 accuracy 0.4906
For gamma 1e-08 penalty(C) 1e-06 accuracy 0.4906
For gamma 1e-08 penalty(C) 1e-05 accuracy 0.4906
For gamma 1e-08 penalty(C) 0.0001 accuracy 0.4906
For gamma 1e-08 penalty(C) 0.001 accuracy 0.4906
For gamma 1e-08 penalty(C) 0.01 accuracy 0.4906
For gamma 1e-08 penalty(C) 0.1 accuracy 0.4906
For gamma 1e-08 penalty(C) 1 accuracy 0.5754
For gamma 1e-08 penalty(C) 10 accuracy 0.6886
For gamma 1e-08 penalty(C) 100 accuracy 0.7012
For gamma 1e-08 penalty(C) 1000 accuracy 0.7102
For gamma 1e-08 penalty(C) 10000 accuracy 0.7116
For gamma 1e-08 penalty(C) 100000 accuracy 0.7218


For gamma 1e-07 penalty(C) 1e-12 accuracy 0.4906
For gamma 1e-07 penalty(C) 1e-11 accuracy 0.4906
For gamma 1e-07 penalty(C) 1e-10 accuracy 0.4906
For gamma 1e-07 penalty(C) 1e-09 accuracy 0.4906
For gamma 1e-07 penalty(C) 1e-08 accuracy 0.4906
For gamma 1e-07 penalty(C) 1e-07 accuracy 0.4906
For gamma 1e-07 penalty(C) 1e-06 accuracy 0.4906
For gamma 1e-07 penalty(C) 1e-05 accuracy 0.4906
For gamma 1e-07 penalty(C) 0.0001 accuracy 0.4906
For gamma 1e-07 penalty(C) 0.001 accuracy 0.4906
For gamma 1e-07 penalty(C) 0.01 accuracy 0.4906
For gamma 1e-07 penalty(C) 0.1 accuracy 0.5392
For gamma 1e-07 penalty(C) 1 accuracy 0.682
For gamma 1e-07 penalty(C) 10 accuracy 0.7008
For gamma 1e-07 penalty(C) 100 accuracy 0.7148
For gamma 1e-07 penalty(C) 1000 accuracy 0.7272
For gamma 1e-07 penalty(C) 10000 accuracy 0.7272
For gamma 1e-07 penalty(C) 100000 accuracy 0.7162


For gamma 1e-06 penalty(C) 1e-12 accuracy 0.4906
For gamma 1e-06 penalty(C) 1e-11 accuracy 0.4906
For gamma 1e-06 penalty(C) 1e-10 accuracy 0.4906
For gamma 1e-06 penalty(C) 1e-09 accuracy 0.4906
For gamma 1e-06 penalty(C) 1e-08 accuracy 0.4906
For gamma 1e-06 penalty(C) 1e-07 accuracy 0.4906
For gamma 1e-06 penalty(C) 1e-06 accuracy 0.4906
For gamma 1e-06 penalty(C) 1e-05 accuracy 0.4906
For gamma 1e-06 penalty(C) 0.0001 accuracy 0.4906
For gamma 1e-06 penalty(C) 0.001 accuracy 0.4906
For gamma 1e-06 penalty(C) 0.01 accuracy 0.4906
For gamma 1e-06 penalty(C) 0.1 accuracy 0.6294
For gamma 1e-06 penalty(C) 1 accuracy 0.7046
For gamma 1e-06 penalty(C) 10 accuracy 0.7338
For gamma 1e-06 penalty(C) 100 accuracy 0.7192
For gamma 1e-06 penalty(C) 1000 accuracy 0.7032
For gamma 1e-06 penalty(C) 10000 accuracy 0.6796
For gamma 1e-06 penalty(C) 100000 accuracy 0.6722


For gamma 1e-05 penalty(C) 1e-12 accuracy 0.4906
For gamma 1e-05 penalty(C) 1e-11 accuracy 0.4906
For gamma 1e-05 penalty(C) 1e-10 accuracy 0.4906
For gamma 1e-05 penalty(C) 1e-09 accuracy 0.4906
For gamma 1e-05 penalty(C) 1e-08 accuracy 0.4906
For gamma 1e-05 penalty(C) 1e-07 accuracy 0.4906
For gamma 1e-05 penalty(C) 1e-06 accuracy 0.4906
For gamma 1e-05 penalty(C) 1e-05 accuracy 0.4906
For gamma 1e-05 penalty(C) 0.0001 accuracy 0.4906
For gamma 1e-05 penalty(C) 0.001 accuracy 0.4906
For gamma 1e-05 penalty(C) 0.01 accuracy 0.4906
For gamma 1e-05 penalty(C) 0.1 accuracy 0.4908
For gamma 1e-05 penalty(C) 1 accuracy 0.7056
For gamma 1e-05 penalty(C) 10 accuracy 0.6942
For gamma 1e-05 penalty(C) 100 accuracy 0.6906
For gamma 1e-05 penalty(C) 1000 accuracy 0.6906
For gamma 1e-05 penalty(C) 10000 accuracy 0.6906
For gamma 1e-05 penalty(C) 100000 accuracy 0.6906


For gamma 0.0001 penalty(C) 1e-12 accuracy 0.4906
For gamma 0.0001 penalty(C) 1e-11 accuracy 0.4906
For gamma 0.0001 penalty(C) 1e-10 accuracy 0.4906
For gamma 0.0001 penalty(C) 1e-09 accuracy 0.4906
For gamma 0.0001 penalty(C) 1e-08 accuracy 0.4906
For gamma 0.0001 penalty(C) 1e-07 accuracy 0.4906
For gamma 0.0001 penalty(C) 1e-06 accuracy 0.4906
For gamma 0.0001 penalty(C) 1e-05 accuracy 0.4906
For gamma 0.0001 penalty(C) 0.0001 accuracy 0.4906
For gamma 0.0001 penalty(C) 0.001 accuracy 0.4906
For gamma 0.0001 penalty(C) 0.01 accuracy 0.4906
For gamma 0.0001 penalty(C) 0.1 accuracy 0.4906
For gamma 0.0001 penalty(C) 1 accuracy 0.5392
For gamma 0.0001 penalty(C) 10 accuracy 0.5612
For gamma 0.0001 penalty(C) 100 accuracy 0.5612
For gamma 0.0001 penalty(C) 1000 accuracy 0.5612
For gamma 0.0001 penalty(C) 10000 accuracy 0.5612
For gamma 0.0001 penalty(C) 100000 accuracy 0.5612


For gamma 0.001 penalty(C) 1e-12 accuracy 0.4906
For gamma 0.001 penalty(C) 1e-11 accuracy 0.4906
For gamma 0.001 penalty(C) 1e-10 accuracy 0.4906
For gamma 0.001 penalty(C) 1e-09 accuracy 0.4906
For gamma 0.001 penalty(C) 1e-08 accuracy 0.4906
For gamma 0.001 penalty(C) 1e-07 accuracy 0.4906
For gamma 0.001 penalty(C) 1e-06 accuracy 0.4906
For gamma 0.001 penalty(C) 1e-05 accuracy 0.4906
For gamma 0.001 penalty(C) 0.0001 accuracy 0.4906
For gamma 0.001 penalty(C) 0.001 accuracy 0.4906
For gamma 0.001 penalty(C) 0.01 accuracy 0.4906
For gamma 0.001 penalty(C) 0.1 accuracy 0.4906
For gamma 0.001 penalty(C) 1 accuracy 0.4926
For gamma 0.001 penalty(C) 10 accuracy 0.4942
For gamma 0.001 penalty(C) 100 accuracy 0.4942
For gamma 0.001 penalty(C) 1000 accuracy 0.4942
For gamma 0.001 penalty(C) 10000 accuracy 0.4942
For gamma 0.001 penalty(C) 100000 accuracy 0.4942


For gamma 0.01 penalty(C) 1e-12 accuracy 0.4906
For gamma 0.01 penalty(C) 1e-11 accuracy 0.4906
For gamma 0.01 penalty(C) 1e-10 accuracy 0.4906
For gamma 0.01 penalty(C) 1e-09 accuracy 0.4906
For gamma 0.01 penalty(C) 1e-08 accuracy 0.4906
For gamma 0.01 penalty(C) 1e-07 accuracy 0.4906
For gamma 0.01 penalty(C) 1e-06 accuracy 0.4906
For gamma 0.01 penalty(C) 1e-05 accuracy 0.4906
For gamma 0.01 penalty(C) 0.0001 accuracy 0.4906
For gamma 0.01 penalty(C) 0.001 accuracy 0.4906
For gamma 0.01 penalty(C) 0.01 accuracy 0.4906
For gamma 0.01 penalty(C) 0.1 accuracy 0.4906
For gamma 0.01 penalty(C) 1 accuracy 0.4906
For gamma 0.01 penalty(C) 10 accuracy 0.4906
For gamma 0.01 penalty(C) 100 accuracy 0.4906
For gamma 0.01 penalty(C) 1000 accuracy 0.4906
For gamma 0.01 penalty(C) 10000 accuracy 0.4906
For gamma 0.01 penalty(C) 100000 accuracy 0.4906


For gamma 0.1 penalty(C) 1e-12 accuracy 0.4906
For gamma 0.1 penalty(C) 1e-11 accuracy 0.4906
For gamma 0.1 penalty(C) 1e-10 accuracy 0.4906
For gamma 0.1 penalty(C) 1e-09 accuracy 0.4906
For gamma 0.1 penalty(C) 1e-08 accuracy 0.4906
For gamma 0.1 penalty(C) 1e-07 accuracy 0.4906
For gamma 0.1 penalty(C) 1e-06 accuracy 0.4906
For gamma 0.1 penalty(C) 1e-05 accuracy 0.4906
For gamma 0.1 penalty(C) 0.0001 accuracy 0.4906
For gamma 0.1 penalty(C) 0.001 accuracy 0.4906
For gamma 0.1 penalty(C) 0.01 accuracy 0.4906
For gamma 0.1 penalty(C) 0.1 accuracy 0.4906
For gamma 0.1 penalty(C) 1 accuracy 0.4906
For gamma 0.1 penalty(C) 10 accuracy 0.4906
For gamma 0.1 penalty(C) 100 accuracy 0.4906
For gamma 0.1 penalty(C) 1000 accuracy 0.4906
For gamma 0.1 penalty(C) 10000 accuracy 0.4906
For gamma 0.1 penalty(C) 100000 accuracy 0.4906

'''