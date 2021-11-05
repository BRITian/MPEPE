# MPEPE

Introduction
====
MPEPE (codonc4) is a prediction method based on deep learning to improve E. coli protein expression. We provide MPEPE's codonc4 model (MPEPE/MODELS-1027/*.h5), prediction example sequence (MPEPE/Example/*.fa) and its result example (MPEPE/Example/2021_11_5_Pred1027/*.res)

System requirement
=====
1. Python 2.7
2. tensorflow 1.15.0
3. keras 2.1.5
4. theano 1.0.5

Quick Start to install the required program
=====
1. Install the python 2.7 from Anaconda https://www.anaconda.com/
2. pip install tensorflow==1.15.0 (python=2.7)
3. pip install keras==2.1.5
4. pip install theano==1.0.5
5. git clone https://github.com/BRITian/MPEPE

Predict the soluble expression of the sequence in E.coli 
====
Put the model folder ((MPEPE/MODELS-1027/), the predicted python file (MPEPE_pred.codonc4.py) and the nucleic acid sequence file (FILE_NAME.fa, or fasta file with any extension) to be predicted in the same directory, and enter the python=2.7 environment to run:

	python MPEPE_pred.codonc4.py FILE_NAME.fa

The prediction result of the final model will be recorded in "Year_Month_Day_Pred1027/Pred_codonc4_FILE_NAME.res" 

Result analysis 
====
In addition to the comment("#") rows, there are three columns. The first column is the IDs of the predicted sequences, the second column is the average value of high expression probability (AVE) predicted by 10 models, and the third column is the average value (AVE) predicted by 10 models that the sequence is Standard deviation of high probability of expression (STD) :

	# === Predicted the probability of highly expressed proteins ===	# (comment row)
	# id	AVE(High_expression)	STD(High_expression)			# (comment row）
	AaR97-5	0.8260	0.0706
	AbR19-5	0.7625	0.1067
	AbR28-5	0.7391	0.1390
	ZR348-5	0.7630	0.0840
	ZR319-5	0.8499	0.0674
	ZR310-5	0.8511	0.0575

As shown in the example (high_expression_seq.fa) results above, the larger the value in the second column (AVE), the better the expression of the sequence in E. coli. 
