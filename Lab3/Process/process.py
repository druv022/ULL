import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('30_Skipgram_1.pkl','rb') as f:
    results_sk= pickle.load(f)

with open('Embedalign.pkl','rb') as f:
    results_em= pickle.load(f)

print("SG: ",results_sk.keys())
print("SG: ", results_sk.values())
print("EA: ",results_em.keys(), "\nEA: ",results_em.values())

other_test = ['STS14']

def get_plot_data(results, x_label=None, inner_labels = None):
    data_1 = []
    data_2 = {}
    x_label1 = []
    x_label2 = []

    if x_label == None:
        get_from = results.keys()
    else:
        get_from = x_label

    for i in get_from:
        print(i,results_sk[i])
        if 'acc' in results[i].keys():
            x_label1.append(i)
            data_1.append(results[i]['acc'])
            # print(i,results[i]['acc'])
        elif i in other_test:
            # print(i)
            # print('BAAM')
            if inner_labels == None:
                get_inner = results[i].keys()
            else:
                get_inner = inner_labels

            for j in get_inner:
                # print("@#@", j)
                x_label2.append(j)
                data_2[j] = results[i][j]
        else:
            print('BOOM')

    return [x_label1, data_1],[x_label2, data_2]

def get_filtered_data(data, keys_seq):
    y_data_p = []
    y_data_c = []
    y_data_ns = []
    for i in data.keys():
        inside_d = data[i]
        # print(inside_d)
        for j in keys_seq:
            if j == 'pearson':
                try:
                    y_data_p.append(inside_d[j][0])
                except :
                    # weighted mean of all types of data
                    y_data_p.append(inside_d[j]['wmean'])

            elif j == 'spearman':
                try:
                    y_data_c.append(inside_d[j][0])
                except :
                    # weighted mean of all types of data
                    y_data_c.append(inside_d[j]['wmean'])
            elif j == 'nsamples' and j in inside_d.keys():
                try:
                    y_data_ns.append(inside_d[j][0])
                except :
                    y_data_ns.append(inside_d[j])
            else:
                pass

    return [y_data_p,y_data_c,y_data_ns]

sk_1, sk_2 = get_plot_data(results_sk)
x_label = sk_1[0]
sk_data = sk_1[1]
x_label2 = sk_2[0]
sk_data2 = sk_2[1]

# print("SK: ", x_label, "\nSK: ", sk_data)
# print("SK: ", x_label2, "\nSK: ", sk_data2)

em_1, dummy = get_plot_data(results_em,x_label)
dummy_ , em_2 = get_plot_data(results_em,other_test,x_label2)

em_data = em_1[1]
em_data2 = em_2[1]

# print("EM: ", x_label, "\nEM: ", em_data)
# print("EM: ", x_label2, "\nEM: ", em_data2)

#
# # print("SG: ", sk_2)
# # print("EA:",em_2)
#
# # print("SG: ", sk_data2.keys(),"\nSG: ", sk_data2.values())
# # print("EA:",em_data2.keys(),"\nEA: ", em_data2.values())
#
keys_seq= ['pearson','spearman','nsamples']

sk_data2_f = get_filtered_data(sk_data2,keys_seq)
em_data2_f = get_filtered_data(em_data2,keys_seq)

# print("SG: ", keys_seq,"\nSG: ", sk_data2_f)
# print("EA:", keys_seq,"\nEA: ", em_data2_f)


fig, ax = plt.subplots()
index = np.arange(len(x_label))
bar_width = 0.15
opacity = 0.8

rst1 = plt.bar(index,sk_data,bar_width,alpha=opacity,color='b',label='Skipgram')
rsl2 = plt.bar(index+bar_width,em_data,bar_width,alpha=opacity,color='r',label='Embedalign')

plt.xlabel('NLP Tasks')
plt.ylabel('Accuracy')
plt.title('Accuracy of different word embeddings on different NLP Tasks')
plt.xticks(index + bar_width,x_label,rotation=15)
plt.legend()

plt.tight_layout()
plt.show()


fig2, ax = plt.subplots()
index = np.arange(len(x_label2))
x_line = np.linspace(0,len(index))
x_top = plt.plot(x_line,[1]*len(x_line),'-',color='black')
x_bottom = plt.plot(x_line,[-1]*len(x_line),'-',color='black')

plt.xlabel('STS14 Task')
plt.xticks(index,x_label2,rotation=15)
plt.ylabel('Pearson')


# print(index.shape)

plt.plot(index,sk_data2_f[0],'*',color='red',label='Skipgram')
plt.plot(index,em_data2_f[0],'.', color='blue',label='Embedalign')
plt.legend()
plt.show()

fig3, ax = plt.subplots()
index = np.arange(len(x_label2))
x_line = np.linspace(0,len(index))
x_top = plt.plot(x_line,[1]*len(x_line),'-',color='black')
x_bottom = plt.plot(x_line,[-1]*len(x_line),'-',color='black')
plt.xlabel('STS14 Task')
plt.xticks(index,x_label2,rotation=15)
plt.ylabel('Spearman Correlation')

plt.plot(index,sk_data2_f[1],'*',color='red',label='Skipgram')
plt.plot(index,em_data2_f[1],'.', color='blue',label='Embedalign')
plt.legend()
plt.show()


