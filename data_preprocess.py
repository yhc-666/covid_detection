import os
import shutil
from sklearn.model_selection import train_test_split

# 划分数据集
covid_dir = '/Users/haochengyang/Desktop/Study/term6/DL/project/dataset/CT_COVID'
noncovid_dir = '/Users/haochengyang/Desktop/Study/term6/DL/project/dataset/CT_NonCOVID'
data_target = []
data_label = []
if not os.path.exists('./covid'):
    os.mkdir('./covid')
if not os.path.exists('./non_covid'):
    os.mkdir('./non_covid')
covid_list = os.listdir(covid_dir)
non_covid_list = os.listdir(noncovid_dir)
for i in range(len(covid_list)):
    img = covid_list[i]
    shutil.copy(covid_dir + '/' + img, './covid/covid' + str(i) + '.png')
    data_target.append('./covid/covid' + str(i) + '.png')
    data_label.append('covid')

for i in range(len(non_covid_list)):
    img = non_covid_list[i]
    shutil.copy(noncovid_dir + '/' + img, './non_covid/non_covid' + str(i) + '.png')
    data_target.append('./non_covid/non_covid' + str(i) + '.png')
    data_label.append('non_covid')

x_train, x_test, y_train, y_test = train_test_split(data_target, data_label, random_state=111, test_size=0.3)
if not os.path.exists('./train_set/covid'):
    os.makedirs('./train_set/covid')
if not os.path.exists('./train_set/non_covid'):
    os.makedirs('./train_set/non_covid')

if not os.path.exists('./test_set/covid'):
    os.makedirs('./test_set/covid')
if not os.path.exists('./test_set/non_covid'):
    os.makedirs('./test_set/non_covid')

for i in range(len(y_train)):
    shutil.copy(x_train[i], './train_set/' + y_train[i])

for i in range(len(y_test)):
    shutil.copy(x_test[i], './test_set/' + y_test[i])

print(x_train[:5])
print(y_train[:5])
