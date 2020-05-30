import pandas as pd
import numpy as np
import DB,LB,Alpha
import tensorflow as tf
import LB
import DB


"""
what you can do with ml:
image classificatio
data clustering
regression
natural language processing

scalar = 1 dimensional variable
vector = multidimensional (each scalar is one dimension)
variable type = usually number, but can also be string
degree/rank = number of dimensions in a tensor. "this text" has dimension 0, ["this text"] has dimension 1 
shape= how many items in each dimension. [2,2] means 2 dimensional, with each 2 data points. similiar like row x col
reshape= different representation of data in matrix
graph = stores all your tensor
session = evaluate graph
type of tensor= variable, constant, placeholder, sparseTensor. all except variable are immutable
evaluate tensor: with tf.Session() as sess; tensor.eval()
batch = if dataset is 25tb, then load data in smaller batches 
epochs= how many times the model uses the same data to train
input function = (x_data,y_data, epochs, batch, shuffle_bool)returns a function to be consumed by model.train
tf.estimator = has all cool premade models like rnn, dnn
evaluate =after making a model, use test data on testing data
linear classification, regression = easiest form 
dense connection = every previous layer is connected to next layer
bias= exist only 1 in previous layer= some numeric constant values
activation function = each layer has its own activation function
relu(rectified linear unit, eliminates negativ enumbers)
softmax()
loss&cost function= a function that assigns a value to describe prediction and actual outcome
mean squred error, mean absolute error 
optimizer=adam= gradient descent= algo to minimize loss function
compile model = choose loss function,optimizer, metric
sequential model = basicl model, passes info from elft to right side
preprocess = scale all values from 255 to between 0 and 1


verify = shows prediction and actual result at same time
supervised:
regression, classification

unsupervised:
clustering



supervised (input + output data: compares output with predict)
predict label could be:
- E/FD/I fgain(freq=1,20,60,240)
- E/FD/I volatility
- RSI when to get normal


unsupervised (input data: creates label/groups)
- let model identify group of labels that have high correlation
- let model identify group of stock hat have correlation
- 

reinforcement (no data: agent, world, goal, the closer to goal, the more reward)
- directly predict close price: reward = gain





Stock ML
Variation 1: predict tomorrow is gain or lose binary
input: [5,10,20,60]x [rsi[], pgain[]]
output: [market bullishness]

Variation 2: predict tomorrow price
input: [5,10,20,60]x [high[], low[]]
output: [market bullishness]

(not very useful since I can manually assign them, not best used as ML)
Variation 3: predict class of a stock [growth vs value],[alpha, beta]
input: [expanding historical data til today]
output: [which class]


"""

df = DB.get_asset()
df = df[['open', 'high', 'low', 'close', 'pct_chg']]
df["tomorrow"] = df['pct_chg'].shift(-1)

x_train = df.head(6000)
y_train = x_train.pop("tomorrow")
x_test = df.tail(600)
y_test = x_test.pop("tomorrow")

feature_columns = []
for column in x_train.columns:
    feature_columns.append(tf.feature_column.numeric_column(column, dtype=tf.float32))


def make_inpuit_fn(data_df, label_df, num_epoch=10, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        ds = ds.batch(batch_size).repeat(num_epoch)
        return ds

    return input_function()


train_input_fn = make_inpuit_fn(x_train, y_train, num_epoch=10)
test_input_fn = make_inpuit_fn(x_test, y_test, num_epoch=1)

linear_estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)
linear_estimator.train(train_input_fn)

result = linear_estimator.evaluate(test_input_fn)
print(result)
