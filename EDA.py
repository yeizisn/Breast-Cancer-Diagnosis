import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

# read and standardization the data
data = pd.read_csv('data.csv')
y = data.diagnosis

x = data.drop(['Unnamed: 32','id','diagnosis'],axis = 1 )
data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())  


#########Figure 1#################
#plot the count for M and B 
sns.set(style="darkgrid", palette="muted")
ax = sns.countplot(y,label="Count",)
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig('./picture/countplot.png',bbox_inches='tight')
plt.close()
B, M = y.value_counts()
print('Number of Benign: ',B)
print('Number of Malignant : ',M)


############Figure 3#####################
#swarm plot
sns.set(style="whitegrid", palette="muted")             # standardization
data = pd.concat([y,data_n_2.iloc[:,0:10]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))

sns.swarmplot(x="features", y="value", hue="diagnosis", data=data,palette="muted")
plt.xticks(rotation=90)
plt.title("mean of features vs. diagnosis")
plt.savefig('./picture/swarm.png',bbox_inches='tight')
plt.show()


########Figure 4###############
# draw one corr fig example
plt.figure()
ax = sns.jointplot(data_n_2.loc[:,'radius_mean'], data_n_2.loc[:,'perimeter_mean'], kind="regg", color="#ca1414")
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.savefig('./picture/jointplot.png')
plt.close()


###########Figure 5######
# draw heatmap
fig = plt.figure(figsize=(15, 10))
sns.heatmap(x.corr(), annot=True, linewidths=.4, fmt= '.1f',cmap="YlGnBu")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
fig.tight_layout()
plt.savefig('./picture/heatmap.png')
plt.close()


#####Figure 6#######
#plot fisher score
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_n_2, y, test_size = 0.3, random_state=42) 
# drop high corr data
high_corr = ['perimeter_mean','radius_mean','compactness_mean','concave points_mean','radius_se',
'perimeter_se','radius_worst','perimeter_worst','compactness_worst','concave points_worst','compactness_se',
'concave points_se','texture_worst','area_worst']
x_train_1 = x_train.drop(high_corr,axis = 1)
x_test_1 = x_test.drop(high_corr,axis = 1)
x1 = data_n_2.drop(high_corr,axis = 1)

# draw Fisher score
fisher = np.zeros(len(x1.columns))
for ind,i in enumerate(x1.columns):
	temp = pd.concat([y,x1[i]],axis=1)
	temp_m = temp[temp.diagnosis=='M']
	temp_b = temp[temp.diagnosis=='B']
	mm = temp_m[i].mean()
	mb = temp_b[i].mean()
	sm = np.mean((temp_m[i]-mm)**2)
	sb = np.mean((temp_b[i]-mb)**2)
	Jw = (mm-mb)**2/(sm+sb)
	fisher[ind] = Jw
fig = plt.figure(figsize=(10,7))
ax = sns.barplot(x=x1.columns,y=fisher,alpha=0.8)
plt.xlabel('features')
plt.ylabel('Fisher Score')
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)
ax.xaxis.label.set_fontsize(15)
ax.yaxis.label.set_fontsize(15)
fig.tight_layout()
plt.savefig(save_path+'fisher_score.png')
plt.close()



















