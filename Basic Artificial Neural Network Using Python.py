# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Install Tensorflow in ANN2
#Install Keras in ANN2
#Install Pandas
#Install scikit lib

#Data Preprocessing

#read the data
import pandas as pd
dataset = pd.read_csv(  "Churn_Modelling.csv"  )

X = dataset.iloc[ :  , 3:13  ].values
Y = dataset.iloc[: , 13:14].values

#Handle missing data# ( part of sklearn)
#[ you will have to handle each and evry column seperatley ...]
#column -0 only : you will create imputer1.
#column - 3 to 12 : you will create  a seperate imputer2.
#..

from sklearn.impute import SimpleImputer
imputer_columnindex0 = SimpleImputer()
X[: , 0:1] = imputer_columnindex0.fit_transform ( X[: , 0:1] )


imputer_columnindex3_9 = SimpleImputer()
X[: , 3: ] =imputer_columnindex3_9.fit_transform( X[: , 3: ] )

#Handle Categorical Data.. ( part of sklearn)
#column = 1 : LE1 , CT1 and OHE1 if required
 #Handle dummpy value trap if you are doning OHE.

from sklearn.preprocessing import LabelEncoder 
labelencoder_X1 = LabelEncoder()
X [: , 1] = labelencoder_X1.fit_transform( X [: , 1] )
 
labelencoder_X2 = LabelEncoder()
X[: , 2] = labelencoder_X2.fit_transform( X[: , 2]  )

#column = 2 : LE2 , CT2 and OHE2 if required
#Handle dummpy value trap if you are doning OHE.

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ohe = OneHotEncoder()
transformer = [   ('encoder', ohe , [1] )     ]

ct = ColumnTransformer( transformer , remainder='passthrough' , sparse_threshold=0 )
X=ct.fit_transform(X)

#avoided dummy value trap
X= X[:, 1:]

import numpy
X = numpy.array(   X , dtype= numpy.float64   )

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X=sc.fit_transform(X)


#training and test set
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest  =train_test_split(  X , Y , test_size=0.2   )


#Ann problem
# first we will create a Sequential Model using keras and tensorflow lib
# train the Model
# use the model to do the prediction.


from keras.models import Sequential
classifer = Sequential()

from keras.layers import Dense

#each dense object is one layer in the ann.
inputlayer = Dense(  output_dim = 4  , init = 'uniform' , activation='relu' ,input_dim = 11 )  


d2 = Dense(  output_dim= 3, init= 'uniform' , activation='relu'  )  
d3 = Dense(  output_dim= 3, init= 'uniform' , activation='relu'  )  
d4 = Dense(  output_dim= 2, init= 'uniform' , activation='relu'  )  

outputlayer = Dense(output_dim=1 , init='uniform' , activation = 'sigmoid')  



#adding the inputy layer first 
classifer.add(  inputlayer)

#adding 1st hidden layer i.e d2 
classifer.add(  d2 )
classifer.add(  d3 )
classifer.add(  d4 )

#adding outputlayer
classifer.add( outputlayer )


#how to optimze the error function ( optimizer )
# what is the error function  ( loss )
#metric to concentrate on ( metric which is a list )
classifer.compile(   optimizer='adam' , loss = 'binary_crossentropy' , metrics=['accuracy'] ) 

#learn
classifer.fit( Xtrain, Ytrain , nb_epoch = 1 , batch_size =2  )
y_pred= classifer.predict( Xtest )

y_pred = ( y_pred > 0.2)



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Ytest, y_pred)

accuracyofmodel = ( ( cm[0,0] + cm[1,1])/ ( cm[0,0] + cm[0,1] + cm [1,0] + cm[1,1]))*100
print("Accuracy of model in the test data is  ", accuracyofmodel)


#new data prediction
#yprednew= classifer.predict(newdata)
yprednew = classifier.predict(newdata)


 

