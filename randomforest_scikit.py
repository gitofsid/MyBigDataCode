"""
/* ******************************************************************************************
 * Name - Siddharth Baronia 
 *
 * Problems - 1. Read training and testng data
 *            2. Using BaggingClassifier train for different sampling rata and forest size
 *            3. Test using validation data
 *            4. Plot accuracy vs penalty for each penalty
 *            5. Repeat all above with feature scaling
 * ********************************************************************************************/ 
"""

import sys, string, math
from pyspark import SparkConf, SparkContext
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

'''sbaronia - draw graphs'''
def draw_graphs(size_arr,accuracy_arr):
    f,ax = plt.subplots()
    colors = ['r','c','k','g','y','m','b','lime','navy','slategrey']
    for i in range(len(accuracy_arr)):
		ax.semilogx(size_arr,accuracy_arr[i], marker='o', linestyle='--', color=colors[i], label='accuracy '+str((i+1)*0.10),linewidth=2)
    plt.xlabel('Size',fontweight='bold')
    plt.ylabel('Accuracy',fontweight='bold')
    plt.legend(loc='best')
    title = "Size vs Accuracy"
    plt.title(title,fontweight='bold')
    plt.show()
    return

def main():

	conf = SparkConf().setAppName('linear kernel SVM')
	sc = SparkContext(conf=conf)
	assert sc.version >= '1.5.1'

	input_path = sys.argv[1]

	data_load = np.load(input_path)

	accuracy_arr = []
	size_arr = [10,20,50,100]
	sample_rate_arr = [i * 0.10 for i in range(1,11)]

	'''sbaronia - for every sample rate run random forest'''
	for sr in sample_rate_arr:
		temp = []
		for si in size_arr:
			bagging = BaggingClassifier(n_estimators=si,max_samples=sr)
			bagging.fit(data_load['data_training'], data_load['label_training'])
			prd = bagging.predict(data_load['data_val']) 
			acc_score = accuracy_score(data_load['label_val'],prd)
			print "For sample rate " + str(sr) + " size " + str(si) + " accuracy " + str(acc_score)
			temp.append(acc_score)
		print "\n"
		accuracy_arr.append(temp)
		
	draw_graphs(size_arr,accuracy_arr)

  
if __name__ == "__main__":
    main()


'''
For sample rate 0.1 size 10 accuracy 0.7586
For sample rate 0.1 size 20 accuracy 0.7754
For sample rate 0.1 size 50 accuracy 0.7796
For sample rate 0.1 size 100 accuracy 0.7862


For sample rate 0.2 size 10 accuracy 0.7876
For sample rate 0.2 size 20 accuracy 0.7982
For sample rate 0.2 size 50 accuracy 0.8026
For sample rate 0.2 size 100 accuracy 0.8076


For sample rate 0.3 size 10 accuracy 0.8016
For sample rate 0.3 size 20 accuracy 0.8132
For sample rate 0.3 size 50 accuracy 0.8218
For sample rate 0.3 size 100 accuracy 0.8212


For sample rate 0.4 size 10 accuracy 0.8134
For sample rate 0.4 size 20 accuracy 0.8252
For sample rate 0.4 size 50 accuracy 0.8328
For sample rate 0.4 size 100 accuracy 0.8306


For sample rate 0.5 size 10 accuracy 0.8208
For sample rate 0.5 size 20 accuracy 0.8304
For sample rate 0.5 size 50 accuracy 0.8412
For sample rate 0.5 size 100 accuracy 0.8424


For sample rate 0.6 size 10 accuracy 0.8216
For sample rate 0.6 size 20 accuracy 0.8332
For sample rate 0.6 size 50 accuracy 0.8458
For sample rate 0.6 size 100 accuracy 0.845


For sample rate 0.7 size 10 accuracy 0.8282
For sample rate 0.7 size 20 accuracy 0.8392
For sample rate 0.7 size 50 accuracy 0.8512
For sample rate 0.7 size 100 accuracy 0.8496


For sample rate 0.8 size 10 accuracy 0.8354
For sample rate 0.8 size 20 accuracy 0.8454
For sample rate 0.8 size 50 accuracy 0.8514
For sample rate 0.8 size 100 accuracy 0.853


For sample rate 0.9 size 10 accuracy 0.8342
For sample rate 0.9 size 20 accuracy 0.8464
For sample rate 0.9 size 50 accuracy 0.8552
For sample rate 0.9 size 100 accuracy 0.8536


For sample rate 1.0 size 10 accuracy 0.8284
For sample rate 1.0 size 20 accuracy 0.8464
For sample rate 1.0 size 50 accuracy 0.8616
For sample rate 1.0 size 100 accuracy 0.855
'''