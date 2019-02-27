import numpy as np
import pandas as pd
from sklearn import tree

col_names = ['mcv', 'alkphos', 'sgpt', 'sgot', 'gammagt', 'drinks', 'selector']
df=pd.read_csv("bupa.csv", header=None,names=col_names)
print(df.shape)
feature = ['mcv', 'alkphos', 'sgpt', 'sgot', 'gammagt', 'drinks']
x=df[feature]   #feature Values
# y is target values
y=df.selector
from sklearn.model_selection import train_test_split
# Split dataset into training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1) # 70% training and 30% test


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(x_train)
x_train_std=sc.transform(x_train)
x_test_std=sc.transform(x_test)


from sklearn.linear_model import LogisticRegression
# Create Logistic Regression classifer object
log = LogisticRegression(C=1,random_state=0,penalty="l2",solver='lbfgs')
# Train Decision Tree Classifer
log.fit(x_train_std,y_train)
#Predict the response for test dataset
y_pred = log.predict(x_test_std)
from sklearn.metrics import accuracy_score
print("Accuracy:",accuracy_score(y_test, y_pred))

#find out best accuracy in deffirent deffirent c values
ma_accuracy=[]
for i in range(1,101):
    log=LogisticRegression(C=i,random_state=0,penalty="l2",solver='lbfgs')
    log.fit(x_train,y_train)
    y_pred=log.predict(x_test)
    a=accuracy_score(y_test,y_pred)
    ma_accuracy.append(a)
print(ma_accuracy,end=" , ") 
print("Maximum Accuracy")
print(max(ma_accuracy))



#create confusion metrix
from sklearn import metrics
cnf_matrix=metrics.confusion_matrix(y_test,y_pred)
print(cnf_matrix)
#this model is binary classification so 2*2 metrix


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')



