#NOTE: bad idea to run this all in one segment, would advise using a
#jupyter notebook instead - which is what I actually used for this project


#import needed modules
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import LabelEncoder #this was a mistake, this is meant for the label, not features
#get our data in
trainset = pd.read_csv("titanic_train.csv")
testset = pd.read_csv("titanic_test.csv")
trainset = trainset.reindex(np.random.permutation(trainset.index))
trainset = trainset.fillna(method = "pad")
testset = testset.fillna(method = "pad")
#build the feed-forward neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(7,)),
    tf.keras.layers.Dense(512,activation="relu"),
    tf.keras.layers.Dense(124,activation="softmax"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1,activation="sigmoid")
])
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
    loss = "binary_crossentropy",
    metrics = ["accuracy"]
)
def transform(value):
    hash = {0:"Z",1:"O",2:"T",3:"P"}
    value = hash[value]
    return value
trainset["Pclass"] = trainset["Pclass"].apply(transform)
encode = LabelEncoder()
trainset["Sex"] = encode.fit_transform(trainset["Sex"])
testset["Sex"] = encode.fit_transform(testset["Sex"])
trainset["Pclass"] = encode.fit_transform(trainset["Pclass"])
testset["Pclass"] = encode.fit_transform(testset["Pclass"])
trainset["Embarked"] = encode.fit_transform(trainset["Embarked"])
testset["Embarked"] = encode.fit_transform(testset["Embarked"])
def logging(value):
    try:
        value = math.log(value,10)
    except:
        value = 0
    return value
def scaling(value):
    value = (value-0.42)/(80-0.42)
    return value
trainset["Fare"],trainset["SibSp"],trainset["Parch"] = trainset["Fare"].apply(logging),trainset["SibSp"].apply(logging),trainset["Parch"].apply(logging)
trainset["Age"] = trainset["Age"].apply(scaling)
testset["Fare"],testset["SibSp"],testset["Parch"] = testset["Fare"].apply(logging),testset["SibSp"].apply(logging),testset["Parch"].apply(logging)
testset["Age"] = testset["Age"].apply(scaling)
ytrain = trainset["Survived"]
xtrain = trainset.drop(columns=["Survived","PassengerId","Name","Ticket","Cabin"])
xtest = testset.drop(columns=["PassengerId","Name","Ticket","Cabin"])
model.fit(xtrain,ytrain,validation_split=0.2,epochs=200)
predictions = model.predict(xtest)
predictions_binary = np.round(predictions).astype(int)
final_predictions = []
for i in predictions_binary:
    for j in i:
        final_predictions.append(j)
output = pd.DataFrame({'PassengerId': testset.PassengerId, 'Survived': final_predictions})
output.to_csv('model_preds.csv', index=False)
