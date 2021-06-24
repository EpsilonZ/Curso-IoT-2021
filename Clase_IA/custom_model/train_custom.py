import sys
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
import tensorflow as tf
import subprocess
from sklearn.metrics import confusion_matrix
from sklearn import metrics


activities = ["laying","standing","sitting","walking","walkingdownstairs","walkingupstairs"]

def save_tflite_model_to_cc_file():

	cmd = "xxd -i model_quantized.tflite > ../ble_sense_arduino/model_data.cpp"
	result_of_cmd = subprocess.Popen(cmd,shell=True)
	return result_of_cmd

def evaluate_classifier(classifier, X_test, y_test):

	pred = classifier.predict(X_test)
	predictions = []
	for i in range(len(pred)):
		predictions.append(pred[i].argmax())
	print(confusion_matrix(y_test,predictions))
	print(str(metrics.accuracy_score(y_test, predictions)*100))

def export_model_to_tflite(model, X_test, y_test):

	#same way as magic wand example https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/examples/magic_wand/train/train.py

	# Convert the model to the TensorFlow Lite format without quantization
	converter = tf.lite.TFLiteConverter.from_keras_model(model)
	tflite_model = converter.convert()

	# Save the model to disk
	open("model.tflite", "wb").write(tflite_model)
		
	def representative_dataset_gen():
		for sample in X_test:
		    sample = np.expand_dims(sample.astype(np.float32), axis=0)
		    yield [sample]
			
	# Convert the model to the TensorFlow Lite format with quantization
	converter = tf.lite.TFLiteConverter.from_keras_model(model)
	converter.optimizations = [tf.lite.Optimize.DEFAULT]
	#converter.target_spec.supported_types = [tf.float32]
	converter.representative_dataset = representative_dataset_gen
		
	tflite_model = converter.convert()
	
	# Save the model to disk
	open("model_quantized.tflite", "wb").write(tflite_model)
	save_tflite_model_to_cc_file()

def train_with_dataset(df):

	y = df['activity'].values.reshape(-1,1)
	onehotencoder = OneHotEncoder()	
	
	df.to_csv('data/blesense_imu_activities.csv', sep=',')	
	
	del df['activity']	
	X = df
	
	classifier = Sequential()
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)	
	
	y_train = onehotencoder.fit_transform(y_train).toarray()	
	
	X_train = np.array([x[1] for x in X_train.iterrows()])
	X_test = np.array([x[1] for x in X_test.iterrows()])
	
	classifier.add(Dense(64, input_dim=128*6, kernel_initializer='uniform', activation='relu', ))
	classifier.add(Dropout(0.3))
	classifier.add(Dense(16))
	classifier.add(Dropout(0.1))
	classifier.add(Dense(8))
	classifier.add(Dense(len(activities), kernel_initializer='uniform', activation='softmax'))
	classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])
	
	print(y_train)
	
	classifier.fit(X_train, y_train, batch_size=20, epochs=25, verbose = 1)

	evaluate_classifier(classifier, X_test, y_test)
	export_model_to_tflite(classifier, X_test, y_test)

def main():
	data_dic = {}
	log_files=glob.glob("activities_data/*")
	df_total = pd.DataFrame()
	for log_file in log_files:
		df_activity = pd.DataFrame()	
		activity = log_file.split("/")[1].split(".")[0]
		data_dic[activity] = {}
		for i in range(384):
			data_dic[activity]["gyro_"+str(i)] = []
			data_dic[activity]["acc_"+str(i)] = []
		imu_gyro_data = []
		imu_acc_data = []
		with open(log_file,'r') as file:
			for line in file:
				try:
					line_data = line.split(" ")
					cont = 0
					line_data = line_data[:-1]	
					if(len(line_data)<300):
						continue
					for info in line_data:
						if(cont<384):
							data_dic[activity]["acc_"+str(cont)].append(float(info))
						else:
							data_dic[activity]["gyro_"+str(cont-384)].append(float(info))
						cont = cont + 1
				except Exception as e:
					print(e)
					pass
								
		#first add acc
		for i in range(384):
			acc_data = []
			for k in data_dic[activity]["acc_"+str(i)]:
				acc_data.append(k)
			df_tmp = pd.DataFrame({"acc_"+str(i): acc_data})
			df_activity = pd.concat([df_activity, df_tmp], axis=1)
			
		for i in range(384):
			acc_data = []
			for k in data_dic[activity]["gyro_"+str(i)]:
				acc_data.append(k)
			df_tmp = pd.DataFrame({"gyro_"+str(i): acc_data})
			df_activity = pd.concat([df_activity, df_tmp], axis=1)
			
		df_act = pd.DataFrame({"activity": [int(activities.index(activity))]*len(df_activity.index) })
		df_activity = pd.concat([df_activity, df_act], axis=1)
		df_total = df_total.append(df_activity)

		print(activity)
		print(df_activity)
		

	print(len(df_total.index))
	
	train_with_dataset(df_total)

if __name__=="__main__":
	main()
