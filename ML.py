import pandas as pd
import numpy as np
import DB,LB,Alpha
import tensorflow as tf

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


"""

print(tf.version)