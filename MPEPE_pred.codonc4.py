# P1  import packages
import os
import sys
import time
import numpy as np
from keras.models import load_model
from keras.preprocessing import sequence
import datetime as dt
import pandas as pd

# P2  define functions


def name_seq(fasta_file):
    with open(fasta_file, 'r') as file:
        lines = file.readlines()
    name_list, seq_list = [], []
    for line in lines:
        if '>' in line:
            name_list.append(line.strip().replace('>', ''))
            seq_list.append('')
        else:
            seq_list[-1] = '%s%s' % (seq_list[-1], line.strip())
    if len(name_list) == 1:
        name_list, seq_list = name_list[0], seq_list[0]
    else:
        pass
    return name_list, seq_list


def nucl2codon(nucl_seq):
    nucl_seq = nucl_seq.upper().replace('U', 'T')
    codon_num, codon_list, aa_list = int(len(nucl_seq) / 3), [], []
    if len(nucl_seq[(codon_num - 1) * 3:]) != 3:
        print('*** Error! => Your sequence is not a cds!!!  Please check your sequence.')
        sys.exit()
    else:
        for i in range(codon_num):
            codon_list.append(nucl_seq[0 + 3 * i: 3 + 3 * i])
            if codon_aa.get(nucl_seq[0 + 3 * i: 3 + 3 * i]) is not None:
                aa_list.append(codon_aa.get(nucl_seq[0 + 3 * i: 3 + 3 * i]))
            else:
                aa_list.append('*')
        if '*' in ''.join(aa_list[:-1]):
            print('## Note! => Stop codons or Unknown amino acids before the end of the sequence. ')
        else:
            pass
    return codon_list


def coding_codon(codon_list):
    coding_list = []
    for item in codon_list:
        if codonc4_coding.get(item) is None:
            coding_list.append('0')
        else:
            coding_list.append(str(codonc4_coding.get(item)))
    return coding_list


def count_ave_std(res_list, step):
    temp_ave, temp_std = [], []
    for s1 in range(step):
        temp_data = res_list[s1::step]
        temp_ave.append(np.average(temp_data))
        temp_std.append(np.std(temp_data))
    return temp_ave, temp_std


# P3  setting parameters
np.set_printoptions(suppress=True)
infile = sys.argv[1]
sp_name = "codonc4"
model_dir = 'MODELs-1027'
pred_dir = '%s_Pred1027' % ('_'.join(np.array(time.localtime(), dtype='str')[:3]))
max_seq_len = 1000
fold = 10

# P4  make dir and coding file
if not os.path.exists(pred_dir):
    os.makedirs(pred_dir)
codon_aa = {'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A', 'TGT': 'C', 'TGC': 'C',
            'GAC': 'D', 'GAT': 'D', 'GAG': 'E', 'GAA': 'E', 'TTC': 'F', 'TTT': 'F',
            'GGG': 'G', 'GGA': 'G', 'GGT': 'G', 'GGC': 'G', 'CAC': 'H', 'CAT': 'H',
            'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'AAG': 'K', 'AAA': 'K', 'CTA': 'L',
            'CTC': 'L', 'CTT': 'L', 'TTG': 'L', 'TTA': 'L', 'CTG': 'L', 'ATG': 'M',
            'AAT': 'N', 'AAC': 'N', 'CCC': 'P', 'CCT': 'P', 'CCA': 'P', 'CCG': 'P',
            'CAA': 'Q', 'CAG': 'Q', 'AGG': 'R', 'AGA': 'R', 'CGG': 'R', 'CGA': 'R',
            'CGT': 'R', 'CGC': 'R', 'AGT': 'S', 'TCG': 'S', 'TCC': 'S', 'TCT': 'S',
            'TCA': 'S', 'AGC': 'S', 'ACT': 'T', 'ACA': 'T', 'ACG': 'T', 'ACC': 'T',
            'GTC': 'V', 'GTA': 'V', 'GTT': 'V', 'GTG': 'V', 'TGG': 'W', 'TAC': 'Y', 'TAT': 'Y'}
codonc4_coding = {'GCT': 1, 'GCC': 2, 'GCA': 3, 'GCG': 4, 'TGT': 5, 'TGC': 6, 'GAC': 7,
                  'GAT': 8, 'GAG': 9, 'GAA': 10, 'TTC': 11, 'TTT': 12, 'GGG': 13, 'GGA': 14,
                  'GGT': 15, 'GGC': 16, 'CAC': 17, 'CAT': 18, 'ATA': 19, 'ATC': 20, 'ATT': 21,
                  'AAG': 22, 'AAA': 23, 'CTA': 24, 'CTC': 25, 'CTT': 26, 'TTG': 27, 'TTA': 28,
                  'CTG': 29, 'ATG': 30, 'AAT': 31, 'AAC': 32, 'CCC': 33, 'CCT': 34, 'CCA': 35,
                  'CCG': 36, 'CAA': 37, 'CAG': 38, 'AGG': 39, 'AGA': 40, 'CGG': 41, 'CGA': 42,
                  'CGT': 43, 'CGC': 44, 'AGT': 45, 'TCG': 46, 'TCC': 47, 'TCT': 48, 'TCA': 49,
                  'AGC': 50, 'ACT': 51, 'ACA': 52, 'ACG': 53, 'ACC': 54, 'GTC': 55, 'GTA': 56,
                  'GTT': 57, 'GTG': 58, 'TGG': 59, 'TAC': 60, 'TAT': 61}
start_name = infile.split('/')[-1].replace(infile.split('/')[-1].split('.')[-1], '')[:-1]
name_list, seq_list = name_seq(infile)
file_coding_url = "./%s_codonc4.txt" % start_name
file_coding = open(file_coding_url, 'w', 0)
for name, seq in zip(name_list, seq_list):
    file_coding.write('0 %s ,%s ,\n' % (name.replace(' ', '-'), " ".join(coding_codon(nucl2codon(seq)))))
file_coding.close()
print('Created "%s"' % file_coding_url)
time.sleep(1)

# P5  Loading coding file data
file_pred_url = '%s/Pred_codonc4_%s.res' % (pred_dir, start_name)
file_pred = open(file_pred_url, 'w', 0)

print('Loading data...')
print("nucl-file: %s\tcodonc4-file: %s" % (infile, file_coding_url))

times1 = dt.datetime.now()

pred_data = pd.read_csv(file_coding_url, index_col=False, header=None)
print(pred_data.shape)

y_test_ori = pred_data[0]
x_test_ori = pred_data[1]
name = y_test_ori
x_test = []
y_test = []
for pi in x_test_ori:
    nr = pi.split(' ')[0:-1]
    ndata = map(int, nr)
    x_test.append(ndata)
x_test = np.array(x_test)

times2 = dt.datetime.now()
print('Time spent: '+ str(times2-times1))

pred_seq=sequence.pad_sequences(x_test, maxlen=max_seq_len)
print(pred_seq)

# P6  predict and record results
all_result0, all_result1, model_use,  = [], [], []
for f1 in range(fold):
    print('# === Fold: %s ===\n' % (f1 + 1))
    model_url = '%s/%s' % (model_dir, 'Best_model_LSTM_R%s_sp%s_rand1027.h5' % ((f1 + 1), sp_name))
    model = load_model(model_url)
    pred_result = model.predict(pred_seq)
    #
    for i in range(len(pred_result)):
        all_result0.append(pred_result[i][0])
        all_result1.append(pred_result[i][1])
    del model
#
ave0, std0 = count_ave_std(all_result0, len(name))
ave1, std1 = count_ave_std(all_result1, len(name))
file_pred.write('# === Predicted the probability of highly expressed proteins ===\n'
                '# id\tAVE(High_expression)\tSTD(High_expression)\n')
for i in range(len(name)):
    file_pred.write('%s\t%.4f\t%.4f\n' % (name[i].split()[1], ave1[i], float(std1[i])))

file_pred.close()
print('Finished!!!')
