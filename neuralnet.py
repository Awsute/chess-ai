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
lrn_rt = 1.0
def safe_sigmoid(x):
    if abs(x) < 700:
        return 1/(1+math.e**-x)
    else:
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
    def random_net(self, out_size):
        t = []
        prev_len = len(self.inputs)
        for i in range(3):
            l = []
            ln = 32
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
    def __init__(self, input_size, hidden):
        self.inputs = [0]*input_size
        self.hidden = hidden
        self.outputs = []
        self.activator = lambda x: safe_sigmoid(x)

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
                a = self.activator(z)
                acls.append(a)
                zs.append(z)
            acs.append(acls)
            self.outputs.append(zs)
        for i in range(len(self.outputs[len(self.outputs)-1])):
            self.outputs[len(self.outputs)-1][i] = self.activator(self.outputs[len(self.outputs)-1][i])
        return self.outputs
    
    def backprop(self, g, y_c):
        
        y = g[len(g)-1]
        
        #print("cost = " + str(cost))
        cost = 0
        dC = 0
        for o in range(len(y_c)):

            for i in range(len(y_c[o])):
                dC += 2/len(y_c)*(y_c[o][i]-y[i])
                cost += 1/len(y_c)*(y_c[o][i]-y[i])**2
        for l in range(len(g)-1, 0, -1):
            for n in range(len(g[l])):
 
                        z = g[l][n]
                        s = self.activator(z)
                        dCdz = dC*s*(1-s)
                        for p in range(len(g[l])):
                            dCdw = dCdz*g[l][p]
                            self.hidden[l-1][n][0][p] -= lrn_rt*dCdw
                            #print("dCdW" +  str(l) + ",  " + str(n) + ", " + str(p) + " = " + str(dCdw))
                            #self.hidden[l-1][n][0][p] = safe_sigmoid(self.hidden[l-1][n][0][p])

                        dCdb = dCdz
                        self.hidden[l-1][n][1] -= lrn_rt*dCdb  
                        #print("dCdb" +  str(l) + ",  " + str(n) + " = " + str(dCdb))
                        #self.hidden[l-1][n][1] = safe_sigmoid(self.hidden[l-1][n][1])
        return
    
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
            

    

#g = Network(69, [])
#g = g.import_from_file("current_ai.json")
#g.hidden = g.random_net(4)
#g.print_net()
#out = g.output([1]*69)
#l = out[len(out)-1]
#print(l)
#g.output_to_file("current_ai.json")
