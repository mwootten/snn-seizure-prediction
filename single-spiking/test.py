import numpy as np
import sys
from brian2 import *

e_const = 2.718282

def TranslateJsonFile(filename, mode = 'data'):
    rfile = open(filename,'r')
    rfiledataset = eval(rfile.read())
    rfile.close()
    
    ansarr = []
    loopnum = len(rfiledataset[0])
    for i in range(loopnum):
        arr = []
        for j in range(4):
            arr += rfiledataset[j][i]
        ansarr.append(arr)
    
    ansarr = np.array(ansarr)
    
    return ansarr

start_scope()
tau = 20*ms

weight_file = open('weight.txt', 'r')
testing_file = 'traditional-testing.json'
dataset = TranslateJsonFile(testing_file, 'data')
num, input_num = dataset.shape
ansset = [0 for i in range(num//2)] + [1 for i in range(num - num//2)]

epoch = num
t_run = 100*ms
t_run_test = 100*ms

threshold_alpha = 2.5

gamma = 1
sigma = 0.5
kappa = -0.4

#=======================================

eqs = '''
dv/dt = ((e_const) ** (-1*ms/tau)- 1) * v / (1*ms) : 1
thrval : 1
'''

Ginput = PoissonGroup(input_num,dataset[0,:]/tau)
hidden_num = 10
Ghidden = NeuronGroup(hidden_num, model = eqs, threshold = 'v > thrval', reset = 'v -= thrval', method = 'exact')
Ghidden.thrval = threshold_alpha * np.sqrt(3 / input_num)
output_num = 2
Goutput = NeuronGroup(output_num, model = eqs, threshold = 'v > thrval', reset = 'v -= thrval', method = 'exact')
Goutput.thrval = threshold_alpha * np.sqrt(3 / hidden_num)

Sih = Synapses(Ginput, Ghidden, model = 'w : 1', on_pre = 'v_post += w')
Sih.connect(condition = True)
Sih.w = 2 * np.sqrt(3 / input_num) * np.random.random(input_num * hidden_num) -  np.sqrt(3 / input_num)

Sho = Synapses(Ghidden, Goutput, model = 'w : 1', on_pre = 'v_post += w')
Sho.connect(condition = True)
Sho.w = 2 * np.sqrt(3 / hidden_num) * np.random.random(hidden_num * output_num) -  np.sqrt(3 / hidden_num)


Shh = Synapses(Ghidden, Ghidden, model = 'w : 1', on_pre = 'v_post += w')
Shh.connect(condition = True)
Shh.w = kappa + np.zeros(hidden_num*hidden_num)
for i in range(hidden_num):
    Shh.w[i + hidden_num*i] = 0

Soo = Synapses(Goutput, Goutput, model = 'w : 1', on_pre = 'v_post += w')
Soo.connect(condition = True)
Soo.w = kappa + np.zeros(output_num*output_num)
for i in range(output_num):
    Shh.w[i + output_num*i] = 0


SpikeMinput = SpikeMonitor(Ginput, None, record = True)

StateMhidden = StateMonitor(Ghidden, 'v', record = True)
SpikeMhidden = SpikeMonitor(Ghidden, 'v', record = True)

StateMoutput = StateMonitor(Goutput, 'v', record = True)
SpikeMoutput = SpikeMonitor(Goutput, 'v', record = True)


wih = np.array(list(map(float, weight_file.readline().split(','))))
who = np.array(list(map(float, weight_file.readline().split(','))))
thidden = np.array(list(map(float, weight_file.readline().split(','))))
toutput = np.array(list(map(float, weight_file.readline().split(','))))


store()

delta_sum = 0
delta_cnt = 0

ans_correct = 0

for i in range(num):
    restore()
    Sih.w = wih
    Sho.w = who
    Ghidden.thrval = thidden
    Goutput.thrval = toutput

    # set test data
    index = i
    ta = dataset[index,:] / tau
    Ginput.rates = ta

    run(t_run_test)
    spike_count = SpikeMoutput.count
    if SpikeMoutput.num_spikes == 0:
        print('%d : dead' %(i))
        continue
    else:
        spike_count /= SpikeMoutput.num_spikes

    ans = np.zeros(2)
    ans[ansset[index]] = 1

    if spike_count[0] > spike_count[1] and ans[0] == 1:
        ans_correct += 1
    elif spike_count[0] < spike_count[1] and ans[1] == 1:
        ans_correct += 1
    
    delta_output = spike_count - ans
    print('%d : Error = ' %(i), abs(delta_output[0]))
    delta_sum += abs(delta_output[0])
    delta_cnt += 1

print('%d of %d samples are correct' %(ans_correct,epoch))
print('%d out of %d tries are dead' %(epoch-delta_cnt,epoch))
print('error rates are %f' %(delta_sum/delta_cnt))
