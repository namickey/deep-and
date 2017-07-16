#coding:utf-8
import numpy
import scipy.special
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time
import copy

class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learnigrate, see):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        numpy.random.seed(see)
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        numpy.random.seed(see*2)
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        self.lr = learnigrate
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

def simu(learning_rate, count, kuri):
    input_nodes = 2
    hidden_nodes = 4
    output_nodes = 2
    inputs_list = [[0.9,0.9],[0.9,-0.9],[-0.9,0.9],[-0.9,-0.9]]
    targets_list = [[1.0,-1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0]]
    tests_list = [[1.0,1.0],[1.0,-1.0],[-1.0,1.0],[-1.0,-1.0]]

    sac = 0.0
    testsize = 2
    trainCount = 1000
    y = []
    for x in range(testsize)[1:]:
        print('neuralNetwork:' + str(int(kuri*x*learning_rate*1000)))
        n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate, int(kuri*x*learning_rate*1300))
        for i in range(count):
            n.train(inputs_list,targets_list)
            if i % trainCount == 0:
                print('train')
                y.append([copy.deepcopy(n.wih), copy.deepcopy(n.who)])
                print([n.wih, n.who])
        for i, t in enumerate(tests_list):
            if numpy.argmax(targets_list[i]) == numpy.argmax(n.query(t)):
                sac += 1.0
        #print('n.who start')
        #print(n.who)
        #print('n.who end')
        #y.append(n.who)
        #y.append([n.wih, n.who])
    #print('y start')
    #print(y)
    #print('y end')
    print(sac / ((testsize-1)*4.0) * 100)
    return y
    #return sac / ((testsize-1)*4.0) * 100

def loopRate(rate):
    print(rate)
    count = 6000
    res = []
    xpl1 = [0,1]
    xpl2 = [0,1,2,3]
    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)
    ax4 = fig.add_subplot(2,2,4)
    for i in range(10)[1:2]:
        sim = simu(rate, count, i)
        res.append(sim)
        for s in sim:
            ax1.plot(xpl1, s[0][0])
            ax2.plot(xpl1, s[0][1])
            ax3.plot(xpl2, s[1][0])
            ax4.plot(xpl2, s[1][1])
            #plt.plot(xpl, s[0])
        #print(sim[0])
    plt.show()
    #print(res)
    return res

#learning_rate = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12]
#learning_rate = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.009, 0.01]
learning_rate = [0.013]
yp = []
xp = []
with Pool(1) as p:
    i = 0
    for y in p.map(loopRate, learning_rate):
        #print(y)
        for yy in y:
            yp.append(yy)
            xp.append(learning_rate[i])
        i += 1
#plt.plot(xp, yp, 'o')
#plt.show()
