import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Read the inputs to dataframe
df1 = pd.read_csv("Images.csv", sep = ';')
df2 = pd.read_csv("EdgeHistogram.csv", sep = ';')

df2.reset_index(inplace=True)#To reset the index of df2 to seperate column of 80 dimensions
df2.index += 1 #To increment the index of all the rows by 1 ,so that index of df2 will also start from 1
df2.drop(['level_0'], axis=1, inplace=True)#To remove imageid column from df2

Column_name=df1.columns[0]#Extract column name from df1
# df1[Column_name].unique() # To check how many unique class is there

from sklearn.preprocessing import LabelEncoder #Inorder to classify all the class name from text format to numeric (encode) 
# Creating a instance of label Encoder(class).
le = LabelEncoder()
# Using .fit_transform function to fit label
# encoder and return encoded label
label = le.fit_transform(df1[Column_name])#fit_transform function will do the conversion from text to numeric
df2['Class']=label# create new column 'Class' in df2 and attaching the numeric values
df1['Class']=label# create new column 'Class' in df1 and attaching the numeric values

#We have 9144 rows converted in to 101 unique rows
df1.drop_duplicates(inplace= True)
df1.reset_index(inplace=True)
df1.drop(['index'], axis=1, inplace=True)

from sklearn.model_selection import train_test_split#train_test_split this function is used to split our data sets into test set and train set
X_train, X_test, y_train, y_test = train_test_split(df2.drop('Class',axis=1),df2['Class'],test_size=0.30,random_state=88)
#In df2 , without target column 'class' only input is required so used this df2.drop('Class',axis=1) and for only target column df2['Class']
#X-Input ,Y- Output

from sklearn.neighbors import KNeighborsClassifier
# Only for single nearest neighbour we can use below three lines
# knn = KNeighborsClassifier(n_neighbors=1)
# knn.fit(X_train,y_train)
# pred = knn.predict(X_test)

error_rate = []#To measure the error in each of the nearest neighbour with variable value 'i'
for i in range(1,40):#Check neighbours from 1 to 40
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)#fit function used to train the data to train set
    pred_i = knn.predict(X_test)# predict function used to predict with test set
    #pred_i is the predicted output from model
    #y_test is the actual output from the given data
    error_rate.append(np.mean(pred_i != y_test))#comparing pred_i and  y_test then appending into errorrate array ,if equal then 0 is added
plt.figure(figsize=(10,6))#drawing canvas with length 10,width 6
plt.title('Error Rate vs. K Value')
plt.xlabel('K')#In x-axis range(1,40)
plt.ylabel('Error Rate')# In Y-axis error_rate
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.show()

from sklearn.metrics import classification_report#to measure the quality of the prediction
print(classification_report(y_test,pred_i))

from sklearn import metrics
print(metrics.accuracy_score(y_test,pred_i))
