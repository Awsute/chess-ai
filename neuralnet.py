import json
import math
from random import random, randint
#[
# ai takes fen input and outputs is ascii of move
#]

#ASCII RANGE (49 , 91)
#chr(ascii_num) = char
#hidden = [[
#   [[w1, w2, w3], bias1],
#   [[w1, w2, w3], bias2]
# ]] 
#hidden[layer][neuron][0] = weights
#hidden[layer][neuron][1] = bias
lrn_rt = 0.25
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
def matrix_dot_product(m1, m2):
    d = 0
    if len(m1)+len(m2) > 0:
        for i in range(0, len(min(m1, m2))):

            d += m1[i]*m2[i]
        return d
    else:
        return 0

def matrix_softmax(mat):
    m = []
    d = 0
    for i in range(len(mat)):
        d += math.e**mat[i]
        m.append(math.e**mat[i])
    
    for i in range(len(m)):
        m[i] /= d
    
    return m
    
def matrix_magnitude(mat):
    return matrix_dot_product(mat, mat)**0.5
def matrix_normalize(mat):
    n = []
    m = matrix_magnitude(mat)
    for i in range(len(mat)):
        n.append(mat[i]/m)
    return n


class Network:

    def __init__(self, input_size, hidden, activator):
        self.inputs = [0]*input_size
        self.hidden = hidden
        self.outputs = []
        self.activator = activator
    
    def random_net(self, out_size):
        t = []
        prev_len = len(self.inputs)
        for i in range(4):
            l = []
            ln = 4
            for o in range(ln):
                w = []
                for j in range(prev_len):
                    w.append((random()+random()-1))
                l.append([w, (random()+random()-1)])
            t.append(l)
            prev_len = ln
        l = []
        for i in range(out_size):
            w = []
            for j in range(prev_len):
                w.append(random()+random()-1)
            l.append([w, random()+random()-1])
        t.append(l)
        return t
    
    def print_net(self):
        print(self.inputs)
        for layer in self.hidden:
            print(layer)
    
    def output(self, inputs):
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
                z = matrix_dot_product(acs[len(acs)-1], g[l][n][0]) + g[l][n][1]
                a = self.activator.fn(z)
                zs.append(z)
                acls.append(a)
            acs.append(acls)
            self.outputs.append(zs)
        for i in range(len(self.outputs[len(self.outputs)-1])):
            self.outputs[len(self.outputs)-1][i] = self.activator.fn(self.outputs[len(self.outputs)-1][i])
        return self.outputs, acs
    
    def backprop(self, y_c, g, acs):
        y = g[len(g)-1]
        #print("cost = " + str(cost))
        dC = 0

        for i in range(len(y_c)):
            dC += 2*(y_c[i]-y[i])
        for l in range(len(g)-1, 0, -1):
            for n in range(len(g[l])):

                        s = self.activator.dfn(g[l][n])
                        dCdz = dC*s
                        for p in range(len(self.hidden[l-1][n][0])):
                            dCdw = dCdz*acs[l-1][p]
                            self.hidden[l-1][n][0][p] += lrn_rt*dCdw
                            #self.hidden[l-1][n][0][p] = safe_sigmoid(self.hidden[l-1][n][0][p])

                        self.hidden[l-1][n][1] += lrn_rt*dCdz
                        #self.hidden[l-1][n][1] = safe_sigmoid(self.hidden[l-1][n][1])
        return self.hidden
    
    def import_from_file(self, path):
        
        with open(path, 'r') as o:
            d = json.load(o)
            self.hidden = d['hidden']
            self.inputs = d['inputs']
        return self
    
    def output_to_file(self, path):
        with open(path, 'w') as f:
            json.dump({'inputs':self.inputs, 'hidden':self.hidden}, f)
        return
            

g = Network(2, [], Activator(lambda x: safe_sigmoid(x), lambda x: safe_sigmoid(x)*(1-safe_sigmoid(x))))
g.hidden = g.random_net(1)
num_correct = 0
num_wrong = 0
for i in range(0, 1800):
    for o in range(0, 144):
        _1 = randint(0, 10)
        _2 = randint(0, 10)
        out, acs = g.output([_1, _2])
        y_c = [(_1+_2)/20]
        y = out[len(out)-1][0]
        print("calculated: " + str(int(y*20+0.5)))
        print("expected:" + str(_1 + _2))
        if int(y*20+0.5) != _1 + _2:
            num_wrong += 1
            g.backprop(y_c, out, acs)
        else:
            num_correct += 1
        print("\n" + str(num_correct) + "-" + str(num_wrong) + "\n")
