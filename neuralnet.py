import json
import math
import numpy as np
from random import random, randint

#NOTES:
#[i1 i2 i3 in 1]
#dot with each:
#	1st hidden layer
#[w11 w21 w31 wn1 b11]
#[w12 w22 w32 wn2 b21]
#[w13 w23 w33 wn3 b23]
#[w1n w2n w3n wnn b2n]
#
#result of dot:
#[h1 h2 h3 hn 1]


#NETWORK HIDDEN:
#network:[
#   for each layer: [
#       for each node: [w1 w2 w3 wn b]
#   ]
#]
class Activator:
    def __init__(self, fn, dfn):
        self.fn = fn
        self.dfn = dfn

def safe_sigmoid(x):
    if abs(x) < 700:
        return 1/(1+math.e**-x)
    else:
        return 1

def drelu(x):
    if x < 0:
        return 0
    elif x >= 0:
        return 1
def relu(m):
    if m<0:
        return 0
    else:
        return m

def matrix_softmax(mat):
    m = []
    d = 0
    for i in range(len(mat)):
        d += math.e**mat[i]
        m.append(math.e**mat[i])
    
    for i in range(len(m)):
        m[i] /= d
    
    return m
    

class Network:

    def __init__(self, input_size, hidden, activator):
        self.inputs = np.array([0]*input_size)
        self.hidden = hidden
        self.outputs = np.array([])
        self.activator = activator
        self.use_bias = 1
    
    def random_net(self, layer_count, layer_size, out_size):
        t = []
        prev_len = len(self.inputs)
        for i in range(layer_count):
            l = []
            for o in range(layer_size):
                w = []
                for j in range(prev_len + 1):
                    w.append(random()+random()-1)
                l.append(w)
            t.append(l)
            prev_len = layer_size
        l = []
        for i in range(out_size):
            w = []
            for j in range(prev_len+1):
                w.append(random()+random()-1)
            l.append(w)
        t.append(l)
        return np.array(t)
    
    def print_net(self):
        print(self.inputs)
        for layer in self.hidden:
            print(layer)
    
    def predict(self, inputs):
        if len(inputs) != len(self.inputs):
            return
        self.outputs = []
        acs = []
        g = self.hidden
        acs.append(inputs)
        self.outputs.append(inputs)
        for l in range(0, len(g)):
            acls = []
            zs = []
            for n in range(0, len(g[l])):
                #print(str(matrix_dot_product(self.outputs[len(self.outputs)-1], g[l][n][0])) + ", " + str(l))
                z = np.dot(np.append(acs[len(acs)-1], self.use_bias), g[l][n])
                a = self.activator.fn(z)
                zs.append(z)
                acls.append(a)
            acs.append(acls)
            self.outputs.append(zs)
        for i in range(len(self.outputs[len(self.outputs)-1])):
            self.outputs[len(self.outputs)-1][i] = self.activator.fn(self.outputs[len(self.outputs)-1][i])
        return self.outputs, acs
    
    def backprop(self, y_c, inputs, lrn_rt):
        g, acs = self.predict(inputs)
        y = g[len(g)-1]
        #print("cost = " + str(cost))
        dC = 0

        for i in range(len(y_c)):
            dC += 2*(y_c[i]-y[i])
        for l in range(len(g)-1, 0, -1):
            for n in range(len(g[l])):
                s = self.activator.dfn(g[l][n])
                dCdz = dC*s
                for p in range(len(self.hidden[l-1][n])-1):
                    dCdw = dCdz*acs[l-1][p]
                    self.hidden[l-1][n][p] += lrn_rt*dCdw

                self.hidden[l-1][n][len(self.hidden[l-1][n])-1] += lrn_rt*dCdz*self.use_bias
        return self.hidden
    
    def import_from_file(self, path):
        #old node format:[[weights], bias]
        #new node format:[weights, bias]
        with open(path, 'r') as o:
            d = json.load(o)
            self.hidden = d['hidden']
            self.inputs = d['inputs']

        return self
    
    def output_to_file(self, path):
        with open(path, 'w') as f:
            json.dump({'inputs':self.inputs, 'hidden':self.hidden}, f)
        return
            
#g = Network(2, [], Activator(lambda x: safe_sigmoid(x), lambda x: safe_sigmoid(x)*(1-safe_sigmoid(x))))
#g.hidden = g.random_net(1, 6, 1)
#
#num_correct = 0
#num_wrong = 0
#for o in range(0, 100):
#    _1 = randint(0, 5)
#    _2 = randint(0, 5)
#    out, acs = g.predict([_1, _2])
#    correct = _1+_2
#    y_c = [(correct)/10]
#    y = out[len(out)-1][0]
#    
#    print("predicted: " + str(int(y*10+0.5)))
#    print("correct: " + str(correct))
#    if int(y*10+0.5) !=correct:
#        num_wrong += 1
#        g.backprop(y_c, out[0], 0.25)
#    else:
#        num_correct += 1
#    print("\n" + str(num_correct) + "-" + str(num_wrong) + "\n")


