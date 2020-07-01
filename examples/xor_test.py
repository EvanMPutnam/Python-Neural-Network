
import sys
sys.path.append("../")
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