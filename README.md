# Mini-Project
# Title: CROP RECOMMENDATION SYSTEM
## Objective
 Our model takes these into account all the necessary factors and recommend the appropriate crop to be grown to give maximum yield and avoid the undesirable results by providing effective solutions using machine learning techniques.

## code

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

df=pd.read_csv('/content/Crop.csv')

df.head()

df.isnull().sum()

df.info()

plt.title("Dataset with outliers")

df.boxplot()

plt.show()

plt.figure(figsize=(5,5))

cols = ['P','K','ph','temperature','humidity','rainfall']

Q1 = df[cols].quantile(0.25)

Q3 = df[cols].quantile(0.75)

IQR = Q3 - Q1

df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

plt.title("Dataset after removing outliers")

df.boxplot()

plt.show()

df.corr()

sns.heatmap(df.corr(),annot=True)

plt.figure(figsize=(15,6))

sns.barplot(y='K',x='label',data=df,palette='hls')

plt.xticks(rotation=90)

plt.show()

x=df.drop('label',axis=1)

y=df['label']

from sklearn.model_selection import  train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 7).fit(x_train, y_train)

knn_predictions = knn.predict(x_test)

accuracy = knn.score(x_test, y_test)

accuracy

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB().fit(x_train, y_train)

gnb_predictions = gnb.predict(x_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, gnb_predictions)

cm

accuracy = gnb.score(x_test, y_test)

accuracy

!pip install gradio

import gradio as gr

def crop(N,P,K,temperature,humidity,ph,rainfall):

  x=np.array([N,P,K,temperature,humidity,ph,rainfall])
  
  prediction = gnb.predict(x.reshape(1,-1))
  
  return prediction
  
 outputs= gr.outputs.Textbox()
 
app=gr.Interface(fn=crop,inputs=['number','number','number','number','number','number','number'],outputs=outputs,description="CROP RECOMMENDATION SYSTEM")

app.launch()

# Code implementation

![image](https://github.com/Thirisaa/Mini-Project/assets/112301582/6b2adc6e-8232-4e15-b3c9-666abc9e84fb)
![image](https://github.com/Thirisaa/Mini-Project/assets/112301582/4cc4386b-4170-45ea-baba-f26fbea0f8bb)
![image](https://github.com/Thirisaa/Mini-Project/assets/112301582/0c3696cb-a2a5-463b-bfe4-b22bc30253ed)
![image](https://github.com/Thirisaa/Mini-Project/assets/112301582/70a5baef-6998-46f6-98de-527b9033c86b)
![image](https://github.com/Thirisaa/Mini-Project/assets/112301582/f242ab90-687f-498b-b39c-9719e09268ba)
![image](https://github.com/Thirisaa/Mini-Project/assets/112301582/03faacf3-76fa-4947-b037-787cb0c1e626)
![image](https://github.com/Thirisaa/Mini-Project/assets/112301582/4b9e54b5-f05b-4ac7-b4bd-6ad52bfcddb2)
![image](https://github.com/Thirisaa/Mini-Project/assets/112301582/eecb898b-5585-4b51-99ca-c417c9f0d4d6)
![image](https://github.com/Thirisaa/Mini-Project/assets/112301582/4740a8a7-efe3-4fb3-a3ca-1a2139f625b3)
![image](https://github.com/Thirisaa/Mini-Project/assets/112301582/e49bff83-55c7-4585-8b07-eaf26ca0bfe6)

![image](https://github.com/Thirisaa/Mini-Project/assets/112301582/709f0984-bc00-4069-994b-9720500edfa7)

![image](https://github.com/Thirisaa/Mini-Project/assets/112301582/4ce5c52f-0441-461d-9d57-6fa50262cfc9)
![image](https://github.com/Thirisaa/Mini-Project/assets/112301582/1e7d2bc6-2f7e-45ad-a9fb-c180f01a41a5)


# Output

![output-crop recommendation](https://github.com/Thirisaa/Mini-Project/assets/112301582/40480fcb-36c2-44e1-b9be-27a440cb7a98)
