# Coding Train Toy Neural Network:
I recently watched a few videos from coding train where a neural network was created and used for a few small projects.  However, I personaly am not a big javascript developer so I decided to port it to python.  The original javascript repository by Daniel Shiffman can be found [here](https://github.com/CodingTrain/Toy-Neural-Network-JS).

## Requirements:
* Python3
* Numpy (Used to simplify the matrix operations)

## Features:
The network itself is fairly simple and only supports a single hidden layer with X number of nodes.  Addittionaly all input and output data should be normalized between 0 and 1.

### Serialization:
There is functionality for serialization and de-serialization of the network.  The code is pretty straightforward.

```python
neural = NeuralNetwork(2, 4, 1)
# ...
# ...
# ...
data = neural.serialize()
neural = NeuralNetwork.deserialize(data)
```

### XOR complete example:
```python
from neural import *

training_data =[{
    "inputs": [0, 0],
    "outputs": [0]
},
{
    "inputs": [0, 1],
    "outputs": [1]
},
{
    "inputs": [1, 0],
    "outputs": [1]
},
{
    "inputs": [1, 1],
    "outputs": [0]
}]

# Create the network and set some params
neural = NeuralNetwork(2, 300, 1)
learning_rate = 0.3
neural.setLearningRate(learning_rate)

# Set number of iterations to train network
total_iterations = 10000

# Train the network
for i in range(0, total_iterations):
    if i % 1000 == 0:
        print((i / total_iterations) * 100, "%")
    data = random.choice(training_data)
    neural.train(data['inputs'], data['outputs'])

# Look at predictions
test1 = [0, 0]
test2 = [0, 1]
test3 = [1, 0]
test4 = [1, 1]

# Here are some results for our testing.
res1 = neural.predict(test1) 
res2 = neural.predict(test2) 
res3 = neural.predict(test3) 
res4 = neural.predict(test4) 

# Should be close to the following [0, 1, 1, 0]
print([res1, res2, res3, res4])
```