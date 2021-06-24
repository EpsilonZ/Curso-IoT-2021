import sys
import matplotlib.pyplot as plt

def main(argv):
	activity = argv[1].split(".")[0]
	imu_gyro_data = [[],[],[]]
	imu_acc_data = [[],[],[]]
	with open(argv[1],'r') as file:
		for line in file:
			try:
				line_data = line.split("||")
				print (line_data)
				cont = 0
				for info in line_data:
					print(info, cont)
					imu_activity = []
					imu_idxyz = info.split(" ")
					imu_activity.append([float(imu_idxyz[1]),float(imu_idxyz[2]),float(imu_idxyz[3])])
					if(len(imu_activity)>0 and cont<128):
						imu_acc_data[0].append(float(imu_idxyz[1]))
						imu_acc_data[0].append(float(imu_idxyz[2]))
						imu_acc_data[0].append(float(imu_idxyz[3]))
					elif(len(imu_activity)>0 and cont<(128*2)):
						imu_acc_data[1].append(float(imu_idxyz[1]))
						imu_acc_data[1].append(float(imu_idxyz[2]))
						imu_acc_data[1].append(float(imu_idxyz[3]))
					elif(len(imu_activity)>0 and cont<(128*3)):
						imu_acc_data[2].append(float(imu_idxyz[1]))
						imu_acc_data[2].append(float(imu_idxyz[2]))
						imu_acc_data[2].append(float(imu_idxyz[3]))
					elif(len(imu_activity)>0 and cont<(128*4)):
						imu_gyro_data[0].append(float(imu_idxyz[1]))
						imu_gyro_data[0].append(float(imu_idxyz[2]))
						imu_gyro_data[0].append(float(imu_idxyz[3]))
					elif(len(imu_activity)>0 and cont<(128*5)):
						imu_gyro_data[1].append(float(imu_idxyz[1]))
						imu_gyro_data[1].append(float(imu_idxyz[2]))
						imu_gyro_data[1].append(float(imu_idxyz[3]))
					elif(len(imu_activity)>0 and cont<(128*6)):
						imu_gyro_data[2].append(float(imu_idxyz[1]))
						imu_gyro_data[2].append(float(imu_idxyz[2]))
						imu_gyro_data[2].append(float(imu_idxyz[3]))
					cont += 3
			except Exception as e:
				print(e)
				pass

	print(imu_acc_data)

	plt.figure()
	
	x_axis_0 = [i for i in range(len(imu_acc_data[0]))]
	x_axis_1 = [i for i in range(len(imu_acc_data[1]))]
	x_axis_2 = [i for i in range(len(imu_acc_data[2]))]
	plt.plot(x_axis_0, imu_acc_data[0], color='blue', label='ACC_X')
	plt.plot(x_axis_1, imu_acc_data[1], color='green', label='ACC_Y')
	plt.plot(x_axis_2, imu_acc_data[2], color='red', label='ACC_Z')

	#x_gyro = [i[0][0] for i in imu_gyro_data]
	#y_gyro = [i[0][1] for i in imu_gyro_data]
	#z_gyro = [i[0][2] for i in imu_gyro_data]
	#x_axis = [i for i in range(len(x_gyro))]
	#plt.plot(x_axis, x_gyro, color='yellow', label='GYRO_X')
	#plt.plot(x_axis, y_gyro, color='brown', label='GYRO_Y')
	#plt.plot(x_axis, z_gyro, color='purple', label='GYRO_Z')

	plt.legend()
	plt.show()

if __name__=="__main__":
	main(sys.argv)
