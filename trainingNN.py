import numpy as np
import csv
import matplotlib.pyplot as plt

'''
The handwritten digits are already converted into an array format which
in each row of the csv file is one handwritten digit.
Current Setup:
    - The neural Network will learn from scratch the trainingData
Neural network:
    - No bias is added
    - Neural Network architechture: 784 : 16 : 16 : 10
        - 784 inputs
        - 16 hidden nodes in hiddenLayer0
        - 16 hidden nodes in hiddenLayer1
        - 10 output nodes
'''
IMAGE_PIXEL = 28 * 28
FILE_NAME = 'trainingData.csv'
MEMORY_FILE = 'memory.csv'
openData = open(FILE_NAME, 'r')
list_of_image = list(csv.reader(openData))
image_number_intial = np.asarray(list_of_image)
intended_output = np.array([[0]])

# This is to visualise the input data that is feed in to the neural network: In this case the numbers that are in different handwriting styles
test_data = np.loadtxt("trainingData.csv", delimiter=",")
fac = 255 * 0.99 + 0.01
test_imgs = np.asfarray(test_data[:, 1:]) / fac
test_labels = np.asfarray(test_data[:, :1])

lr = np.arange(10)

node_delta = np.empty((0,0))
learning_rate = 0.01

# Initialise the weights
weight0 = 2*np.random.rand(IMAGE_PIXEL, 16)-1
weight1 = 2*np.random.rand(16, 16)-1
weight2 = 2*np.random.rand(16, 16)-1
weight3 = 2*np.random.rand(16, 10)-1

'''
# This is to extract the weights from the csv file
# Established all the connections that the neural net has "learnt"
def extraction(filename='memory.csv'):
    # This acts as memory
    file = open(filename)
    weights = csv.reader(file)
    list_of_weights = list(weights)
    tempExtraction = []
    for index in range(int(len(list_of_weights)/2)):
        position = index * 2
        placeHolder = []
        fullTempArr = []
        tempArr0 = []
        tempArr1 = []
        for num in list_of_weights[position]:
            try:
                tempArr0.append(float(num))
            except:
                tempArr0.append(str(num))
        for index in tempArr0:
            if index == 'zzz':
                fullTempArr.append(tempArr1)
                tempArr1 = []
            else:
                tempArr1.append(index)
        tempExtraction.append(fullTempArr)

    synapse0 = tempExtraction[0]
    synapse1 = tempExtraction[1]
    synapse2 = tempExtraction[2]
    synapse3 = tempExtraction[3]

    return np.asarray(synapse0), np.asarray(synapse1), np.asarray(synapse2), np.asarray(synapse3)
'''
def convertStrArr2NumArr(series):
    tempArr = []
    newArr = []
    for item in series:
        tempArr.append(int(item))
    for element in tempArr:
        newArr.append(float(element))
    return np.asarray(newArr)
# activation function
def sigmoidFunc(x, deriv=False):
    if deriv == True:
        return x*(1-x)
    return 1/(1+np.exp(-x))

print("total of images to train:", str(int(len(list_of_image)/2)))

'''
# These are preinitialised weights that have been extracted from the weights file
w0, w1, w2, w3 = extraction(filename='memory.csv')
weight0 = w0
weight1 = w1
weight2 = w2
weight3 = w3
'''
for index in range(int(len(list_of_image)/2)):
    #img = 2 * index
    input = convertStrArr2NumArr(image_number_intial[index][:])[1:]# Input is a one dimension array
    # changing the array to do supervise the learning process | This is also the validation process as well
    if convertStrArr2NumArr(image_number_intial[index][:])[:1] == 0:
        intended_output = np.asarray([[1], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
    elif convertStrArr2NumArr(image_number_intial[index][:])[:1] == 1:
        intended_output = np.asarray([[0], [1], [0], [0], [0], [0], [0], [0], [0], [0]])
    elif convertStrArr2NumArr(image_number_intial[index][:])[:1] == 2:
        intended_output = np.asarray([[0], [0], [1], [0], [0], [0], [0], [0], [0], [0]])
    elif convertStrArr2NumArr(image_number_intial[index][:])[:1] == 3:
        intended_output = np.asarray([[0], [0], [0], [1], [0], [0], [0], [0], [0], [0]])
    elif convertStrArr2NumArr(image_number_intial[index][:])[:1] == 4:
        intended_output = np.asarray([[0], [0], [0], [0], [1], [0], [0], [0], [0], [0]])
    elif convertStrArr2NumArr(image_number_intial[index][:])[:1] == 5:
        intended_output = np.asarray([[0], [0], [0], [0], [0], [1], [0], [0], [0], [0]])
    elif convertStrArr2NumArr(image_number_intial[index][:])[:1] == 6:
        intended_output = np.asarray([[0], [0], [0], [0], [0], [0], [1], [0], [0], [0]])
    elif convertStrArr2NumArr(image_number_intial[index][:])[:1] == 7:
        intended_output = np.asarray([[0], [0], [0], [0], [0], [0], [0], [1], [0], [0]])
    elif convertStrArr2NumArr(image_number_intial[index][:])[:1] == 8:
        intended_output = np.asarray([[0], [0], [0], [0], [0], [0], [0], [0], [1], [0]])
    elif convertStrArr2NumArr(image_number_intial[index][:])[:1] == 9:
        intended_output = np.asarray([[0], [0], [0], [0], [0], [0], [0], [0], [0], [1]])
    else:
        pass


    print(convertStrArr2NumArr(image_number_intial[index][:])[:1])
    print("Trainning image:", str(index+1), "| Number of images left", str(int((len(list_of_image)/2)-(index+1))))
    ###vv Trainning Algorithm vv####
    for i in range(200):
        #Trains the neural network
        new_input = []
        delta_weights_on_outputLayer = []
        total_delta_hiddenLayer2_output = []
        total_delta_hiddenLayer1_output = []
        total_delta_hiddenLayer0_output = []
        delta_errors = []

        # Feed forward algorithm
        hiddenLayer0 = sigmoidFunc(np.dot(input, weight0))
        hiddenLayer1 = sigmoidFunc(np.dot(hiddenLayer0, weight1))
        hiddenLayer2 = sigmoidFunc(np.dot(hiddenLayer1, weight2))
        outputLayer = sigmoidFunc(np.dot(hiddenLayer2, weight3))

        # This is backpropagation process
        # This is to create the error and minimise the cost function -> this is to find either the local or global minima
        for i in range(len(outputLayer)):
            #item = (intended_output[i] - outputLayer[i])**2 # This is the cost function
            delta_item = -(intended_output[i] - outputLayer[i]) # the derivative from the mean square error cost function
            delta_errors.append(delta_item)

        for _iter in range(len(delta_errors)):
            _tempArr = []
            for _nodes in range(len(hiddenLayer2)):
                node_delta = (-delta_errors[_iter]*sigmoidFunc(outputLayer[_iter], True))*hiddenLayer2[_nodes]#using the chain rule to do backpropagation
                _tempArr = np.append(_tempArr, node_delta)
            delta_weights_on_outputLayer.append(np.array(_tempArr))

        delta_hiddenLayer2_output = np.asarray(delta_weights_on_outputLayer).T*weight3

        for _row in range(len(hiddenLayer2)):
            tempVar = 0
            for _col in range(len(outputLayer)):
                tempVar += delta_hiddenLayer2_output[_row, _col]
            total_delta_hiddenLayer2_output.append(np.asarray(tempVar))
        delta_weights_on_hiddenLayer2 = (-np.asarray(total_delta_hiddenLayer2_output)*sigmoidFunc(hiddenLayer2, True))*hiddenLayer1

        delta_hiddenLayer1_output = np.asarray(delta_weights_on_hiddenLayer2).T*weight2
        for _row in range(len(hiddenLayer1)):
            tempVar = 0
            for _col in range(len(hiddenLayer2)):
                tempVar += delta_hiddenLayer1_output[_row, _col]
            total_delta_hiddenLayer1_output.append(np.asarray(tempVar))
        delta_weights_on_hiddenLayer1 = (-np.asarray(total_delta_hiddenLayer1_output)*sigmoidFunc(hiddenLayer1, True))*hiddenLayer0

        delta_hiddenLayer0_output = np.asarray(delta_weights_on_hiddenLayer1).T*weight1

        for _row in range(len(hiddenLayer0)):
            tempVar = 0
            for _col in range(len(hiddenLayer1)):
                tempVar += delta_hiddenLayer0_output[_row, _col]
            total_delta_hiddenLayer0_output.append(np.asarray(tempVar))

        # transforms the input nodes into a 16, 784
        for iter in range(len(hiddenLayer0)):
            new_input.append(input)
        delta_weights_on_hiddenLayer0 = (-np.asarray(total_delta_hiddenLayer0_output)*sigmoidFunc(hiddenLayer0, True))*np.asarray(new_input).T

        # Gradient descent
        weight3 -= -learning_rate*np.asarray(delta_weights_on_outputLayer).T
        weight2 -= -learning_rate*np.asarray(delta_weights_on_hiddenLayer2).T
        weight1 -= -learning_rate*np.asarray(delta_weights_on_hiddenLayer1).T
        weight0 -= -learning_rate*np.asarray(delta_weights_on_hiddenLayer0)

    print("\nError:", str(np.mean(np.abs(np.asarray(delta_errors)))))# get the average error for each iteration
    print(str(outputLayer), '\n')

    # This is to shows a visualisation of the handwritten digits
    images = test_imgs[index].reshape((28,28))
    plt.imshow(images, cmap="Greys")
    plt.show(block=False)
    plt.pause(0.5)
    plt.close()

strength_of_connection = [np.asarray(weight0), np.asarray(weight1), np.asarray(weight2), np.asarray(weight3)]

with open(MEMORY_FILE, 'a') as data:
    writer = csv.writer(data)
    for item in strength_of_connection:
        tempArr = []
        _row, _col = item.shape
        for x in range(_row):
            for y in range(_col):
                tempArr.append(item[x, y])
            tempArr.append('zzz')
        writer.writerow(tempArr)
