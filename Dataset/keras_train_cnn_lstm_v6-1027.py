# This Python file uses the following encoding: utf-8
"""
Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
"""
# from __future__ import print_function
# import tensorflow
# from ..externals import six
# from ..utils.fixes import in1d
import numpy as np
import theano
import pandas as pd
import shutil, os, sys
import tensorflow as tf
import random
from keras.models import Sequential
from keras.models import load_model
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Flatten, Activation,Embedding, GRU, RepeatVector,SpatialDropout1D,TimeDistributed
from keras.layers import Conv1D, MaxPooling1D,LSTM,BatchNormalization, GlobalMaxPooling1D
from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.preprocessing import sequence
from keras.constraints import maxnorm
from sklearn.metrics import classification_report
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import roc_auc_score
import datetime as dt
from keras import regularizers
from keras.utils import plot_model

sp_name=sys.argv[1]
rand_id = '1027'  # sys.argv[2]
rep_num = 10
max_seq_len = 1000

# gpu_id = '0'
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
# config = tf.ConfigProto()
# config.allow_soft_placement = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.7
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)

theano.config.openmp = True
np.set_printoptions(threshold=sys.maxsize)

nb_epoch = 40
num_classes = 2
# epochs = 20

mulu_train="coding-%s/" % rand_id
mulu_logs="LOGs-%s/" % rand_id
mulu_models="MODELs-%s/" % rand_id
if not os.path.exists(mulu_models):
	os.makedirs(mulu_models)
if not os.path.exists(mulu_logs):
	os.makedirs(mulu_logs)

Dropout_0= 0.2183113053472918
Dropout_1 = 0.20248769159548086
Dropout_2 = 0.24228348711230202
Dropout_3 = 0.258994401132728
Dropout_4 = 0.23996085010756082
Dropout_5 = 0.18388302373625715
filter1 = 64
filter1_1 = 128
kernel_size1 = 15
kernel_size1_1 = 9
l2 = 0.030735773575039663
l2_1 = 0.0340372392694714
pool_size1 = 2
pool_size1_1 = 4
LSTM1 = 64
optimizers = 'adam'
batch_sizes = 64
fea_num = 21

if sp_name == "aac3":
	Dropout_0= 0.2183113053472918
	Dropout_1 = 0.20248769159548086
	Dropout_2 = 0.24228348711230202
	Dropout_3 = 0.258994401132728
	Dropout_4 = 0.23996085010756082
	Dropout_5 = 0.18388302373625715
	filter1 = 64
	filter1_1 = 128
	kernel_size1 = 15
	kernel_size1_1 = 9
	l2 = 0.030735773575039663
	l2_1 = 0.0340372392694714
	pool_size1 = 2
	pool_size1_1 = 4
	LSTM1 = 64
	optimizers = 'adam'
	batch_sizes = 64
	fea_num = 21
	#optimizer=Adam()
if sp_name == "codonc4":
	Dropout_0= 0.30983805961043087
	Dropout_1 = 0.3614095887243577
	Dropout_2 = 0.12306710587932396
	Dropout_3 = 0.2222015642041399
	Dropout_4 = 0.10119930831696444
	Dropout_5 = 0.3184011451831595
	filter1 = 64
	filter1_1 = 128
	kernel_size1 = 15
	kernel_size1_1 = 3
	l2 = 0.05719472998137672
	l2_1 = 0.0696230987109973
	pool_size1 = 2
	pool_size1_1 = 2
	LSTM1 = 64
	optimizers = 'adam'
	batch_sizes = 128
	fea_num = 62
	#optimizer=Adam()
if sp_name == "codonc3":
	Dropout_0= 0.1419426566393742
	Dropout_1 = 0.24977300602965716
	Dropout_2 = 0.17800590944108585
	Dropout_3 = 0.10552203632180934
	Dropout_4 = 0.26413497399882224
	Dropout_5 = 0.24523743017717525
	filter1 = 64
	filter1_1 = 128
	kernel_size1 = 3
	kernel_size1_1 = 3
	l2 = 0.0016648553057593664
	l2_1 = 0.0017050611587937307
	pool_size1 = 2
	pool_size1_1 = 2
	LSTM1 = 64
	optimizers = 'rmsprop'
	batch_sizes = 128
	fea_num = 7
	#optimizer=Adam()
print(Dropout_0)


def shuffleData(X, y):
	index = [i for i in range(len(X))]
	random.shuffle(index)
	X = X[index]
	y = y[index]
	return X, y


for tar_r in range(rep_num):
	tar_rep_id =tar_r + 1
	print(tar_rep_id)
	tar_prefix= "LSTM_R" + str(tar_rep_id) + "_sp" + sp_name + "_rand" + rand_id
	tar_log_file=mulu_logs + "LOG_cnn_" + tar_prefix
	fout=open(tar_log_file, 'w', 0)
	#aa_num=wi+1

	print('Loading data...')
	infile_train=mulu_train + "Train_num_"+ str(sp_name)+  "_" +  str(tar_rep_id)
	infile_test=mulu_train + "Test_num_" + str(sp_name)+ "_" +  str(tar_rep_id) 
	print(infile_train)
	print(infile_test)

	times1 = dt.datetime.now()

	train_data = pd.read_csv(infile_train, index_col = False, header=None)
	test_data = pd.read_csv(infile_test, index_col = False, header=None)
	print(train_data.shape)
	print(test_data.shape)

	y_test_ori=test_data[0]
	x_test_ori=test_data[1]
	x_test=[]
	y_test=[]
	for pi in x_test_ori:
		nr=pi.split(' ')[0:-1]
		ndata=map(int,nr)
		x_test.append(ndata)
	x_test=np.array(x_test)
	for pi in y_test_ori:
		nr=pi.split(' ')[0:-1]
		ndata=int(nr[0])
		y_test.append(ndata)
	y_test=np.array(y_test)
	#
	y_train_ori=train_data[0]
	x_train_ori=train_data[1]
	x_train=[]
	y_train=[]
	for pi in x_train_ori:
		nr=pi.split(' ')[0:-1]
		ndata=map(int,nr)
		x_train.append(ndata)
	x_train=np.array(x_train)
	for pi in y_train_ori:
		nr=pi.split(' ')[0:-1]
		ndata=int(nr[0])
		y_train.append(ndata)
	y_train=np.array(y_train)
	print(x_train.shape)
	#sys.exit()

	times2 = dt.datetime.now()
	print('Time spent: '+ str(times2-times1))

	# convert class vectors to binary class matrices
	#print y_test

	y_train = np_utils.to_categorical(y_train, num_classes)
	y_test = np_utils.to_categorical(y_test, num_classes)
	y_real = np.argmax(y_test, axis=1)
	x_train=sequence.pad_sequences(x_train,maxlen=max_seq_len)
	x_test=sequence.pad_sequences(x_test,maxlen=max_seq_len)
	print(x_train.shape)

	for ki in range(1):
		#tar_kernel= ki *2 +3
		model_filepath= mulu_models + "Best_model_" + tar_prefix + ".h5"

		model = Sequential()
		model.add(Embedding(fea_num, fea_num, input_length=max_seq_len))
		model.add(BatchNormalization())
		model.add(Dropout(Dropout_0))

		model.add(Conv1D(filters=filter1, kernel_size=kernel_size1,
						 strides=1, kernel_regularizer=regularizers.l2(l2),
						 padding='same', kernel_initializer='random_uniform', activation='relu',
						 kernel_constraint=maxnorm(3), bias_constraint=maxnorm(3)))
		model.add(BatchNormalization())
		model.add(Dropout(Dropout_1))

		model.add(MaxPooling1D(pool_size=pool_size1))
		model.add(BatchNormalization())
		model.add(Dropout(Dropout_2))

		model.add(Conv1D(filters=filter1_1, kernel_size=kernel_size1_1,
						 strides=1, kernel_regularizer=regularizers.l2(l2_1),
						 padding='same', kernel_initializer='random_uniform', activation='relu',
						 kernel_constraint=maxnorm(3), bias_constraint=maxnorm(3)))
		model.add(BatchNormalization())
		model.add(Dropout(Dropout_3))

		model.add(MaxPooling1D(pool_size=pool_size1_1))
		model.add(BatchNormalization())
		model.add(Dropout(Dropout_4))

		model.add(LSTM(LSTM1))
		model.add(BatchNormalization())
		model.add(Dropout(Dropout_5))

		model.add(Dense(num_classes, activation='sigmoid'))
		print(model.summary())
		# plot_model(model,to_file='model_auth_%s_%s.png' % (sp_name, tar_rep_id),show_shapes=True)
		# plot_model(model, to_file='model_auth_%s_%s.pdf' % (sp_name, tar_rep_id), show_shapes=True)

		checkpoint = ModelCheckpoint(filepath=model_filepath, monitor='val_acc',
									 verbose=1, save_best_only=True, mode='max')
		callbacks_list = [checkpoint]

		print('Training')
		#model = load_model("Best_model_CNN_template.h5")
		for re in range(nb_epoch):
		#for i in range(20):
			tar_re=re+1
			print('Epoch', tar_re, '/', nb_epoch)

			#OPTIMIZER=RMSprop()
			OPTIMIZER=Adam(amsgrad=True)
			x_train, y_train = shuffleData(x_train,y_train)
			model.compile(loss='binary_crossentropy', metrics=['accuracy'],
						  optimizer=optimizers)

			result = model.fit(x_train, y_train, epochs=20, callbacks=callbacks_list,
							   batch_size=batch_sizes, validation_data=(x_test, y_test),
							   shuffle=True, class_weight='auto', verbose=0)

			loss_and_metrics_train = model.evaluate(x_train, y_train)
			loss_and_metrics_test = model.evaluate(x_test, y_test)
			print >>fout, "Train_Test " + str(tar_re) +" metrics ",loss_and_metrics_train, loss_and_metrics_test

			print loss_and_metrics_train
			print loss_and_metrics_train[1]

		del model
		model = load_model(model_filepath)
		loss_and_metrics = model.evaluate(x_test, y_test)

		y_pred_ori = model.predict(x_test)
		y_pred = np.argmax(y_pred_ori, axis=1) # Convert one-hot to index
	
		print >>fout, "Sp" + "_"+str(sp_name)
		print >>fout, "Final_metrics_Test : ",loss_and_metrics

		loss_and_metrics_train = model.evaluate(x_train, y_train)
		print >>fout, "Final_metrics_Train : ",loss_and_metrics_train,"\n\n"

		print >>fout, classification_report(y_real, y_pred,digits=4)
	
		auc=roc_auc_score(y_test.flatten(),y_pred_ori.flatten())
		print >>fout, "AUC"
		print >>fout, auc

	fout.close()

