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
# inital setting
training_file = 'traditional-training.json'
dataset = TranslateJsonFile(training_file, 'data')
num_samples, input_num = dataset.shape

ansset = [0 for i in range(num_samples//2)] + [1 for i in range(num_samples - num_samples//2)]

tau = 20*ms

testing_file = 'traditional-testing.json'
test_dataset = TranslateJsonFile(testing_file, 'data')
test_num_samples = test_dataset.shape[0]
test_ansset = [0 for i in range(test_num_samples//2)] + [1 for i in range(test_num_samples - test_num_samples//2)]

learn_epoch = 10
batch_size = 10
test_epoch = 100
t_run = 64*ms
t_run_test = 128*ms

learning_rate_weight = 0.007
learning_rate_thrval = 0.03 * learning_rate_weight

threshold_alpha = 2

gamma = 1
sigma = 0.5
kappa = 0

recording_count_read = open('record_count.txt','r')
test_count = eval(recording_count_read.readline()) + 1
recording_count_read.close()
recording_count_write = open('record_count.txt','w')
recording_count_write.write(str(test_count))
recording_count_write.close()

recording_log = open('record_log.txt','a')
recording_log.write('===================================' + '\n')
recording_log.write(str(test_count) + ' test' + '\n')
recording_log.write('trainging file : ' + training_file + '\n')
recording_log.write('testing file : ' + testing_file + '\n')
recording_log.write('learn epoch : ' + str(learn_epoch) + '\n')
recording_log.write('batch size : ' + str(batch_size) + '\n')
recording_log.write('test epoch : ' + str(test_epoch) + '\n')
recording_log.write('t_run(training) : ' + str(t_run/ms) + ' ms' + '\n')
recording_log.write('t_run(testing) : ' + str(t_run_test/ms) + ' ms' + '\n')
recording_log.write('learning rate (weight) : ' + str(learning_rate_weight) + '\n')
recording_log.write('learning rate (thrval) : ' + str(learning_rate_thrval) + '\n')
recording_log.write('threshold_alpha : ' + str(threshold_alpha) + '\n')
recording_log.write('gamma : ' + str(gamma) + '\n')
recording_log.write('sigma : ' + str(sigma) + '\n')
recording_log.write('kappa : ' + str(kappa) + '\n')
recording_log.close()

memo1_name = 'record_' + str(test_count) + '_error.csv'
memo2_name = 'record_' + str(test_count) + '_weight.csv'

file_memo1 = open(memo1_name, 'w')
file_memo2 = open(memo2_name, 'w')

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

'''
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
'''

SpikeMinput = SpikeMonitor(Ginput, None, record = True)

StateMhidden = StateMonitor(Ghidden, 'v', record = True)
SpikeMhidden = SpikeMonitor(Ghidden, 'v', record = True)

StateMoutput = StateMonitor(Goutput, 'v', record = True)
SpikeMoutput = SpikeMonitor(Goutput, 'v', record = True)

# del_hidden = np.zero(hidden_num)
# del_output = np.zero(output_num)

wih = np.array(Sih.w)
who = np.array(Sho.w)
thidden = np.array(Ghidden.thrval)
toutput = np.array(Goutput.thrval)

store()
print('Starting learning...')
file_memo1.write('epoch, error, average thrval\n')
for i in range(learn_epoch):
    if i == 0 or (i+1) % 10 == 0:
        print('Epoch %d beginning!' % (i+1))

        file_memo2.write(str(i+1) + '\n')
        file_memo2.write(','.join(map(str, wih)) + '\n')
        file_memo2.write(','.join(map(str, who)) + '\n')
        file_memo2.write(','.join(map(str, thidden)) + '\n')
        file_memo2.write(','.join(map(str, toutput)) + '\n\n')

    del_wih = np.zeros([input_num, hidden_num])
    del_who = np.zeros([hidden_num, output_num])
    del_hidden_thrval = np.zeros([hidden_num])
    del_output_thrval = np.zeros([output_num])

    error = 0
    for _ in range(batch_size):
        restore()

        Sih.w = wih
        Sho.w = who
        Ghidden.thrval = thidden
        Goutput.thrval = toutput
        
        # set training data
        index = np.random.randint(num_samples)
        ta = dataset[index,:] / tau
        Ginput.rates = ta

        run(t_run)
        spike_count = SpikeMoutput.count
        if SpikeMoutput.num_spikes == 0:
            spike_count = [0.5,0.5]
        else:
            spike_count /= SpikeMoutput.num_spikes

        ans = np.zeros(2)
        ans[ansset[index]] = 1

        delta_output = spike_count - ans
        error += abs(delta_output[0]) / batch_size
        w_ho = np.reshape(Sho.w, [hidden_num,output_num])    
        delta_hidden = np.sqrt(hidden_num / output_num) * np.multiply(1/Ghidden.thrval, (w_ho.dot(delta_output)))
        
        a_input  = SpikeMinput.count
        # x_hidden = Ghidden.thrval * SpikeMhidden.count + StateMhidden.v[:,-1]
        a_hidden = SpikeMhidden.count + StateMhidden.v[:,-1] / Ghidden.thrval
        # x_output = Goutput.thrval * SpikeMoutput.count + StateMoutput.v[:,-1]
        a_output = SpikeMoutput.count + StateMoutput.v[:,-1] / Goutput.thrval

        del_wih += - learning_rate_weight * delta_hidden * np.atleast_2d(a_input).T / batch_size
        del_hidden_thrval += - learning_rate_thrval * delta_hidden * ((gamma+sigma)*a_hidden - sigma * kappa * a_hidden) / batch_size

        del_who += - learning_rate_weight * delta_output * np.atleast_2d(a_hidden).T / batch_size
        del_output_thrval += - learning_rate_thrval * delta_output * ((gamma+sigma)*a_output - sigma * kappa * a_output) / batch_size

    print('Epoch #%d: ' % (i+1), error)

    file_memo1.write(str(i+1) + ', ' + str(error) + ', ' + str((sum(np.array(Ghidden.thrval)) + sum(np.array(Goutput.thrval))) / (hidden_num+output_num)) + '\n')

    wih += np.reshape(del_wih, [-1])
    who += np.reshape(del_who, [-1])

    thidden += del_hidden_thrval
    toutput += del_output_thrval

test_delta_sum = 0
test_delta_cnt = 0

test_ans_correct = 0

for i in range(test_epoch):
    restore()
    Sih.w = wih
    Sho.w = who
    Ghidden.thrval = thidden
    Goutput.thrval = toutput

    # set test data
    index = np.random.randint(test_num_samples)
    ta = test_dataset[index,:] / tau
    Ginput.rates = ta

    run(t_run_test)
    spike_count = SpikeMoutput.count
    if SpikeMoutput.num_spikes == 0:
        print('%d : dead' %(i))
        continue
    else:
        spike_count /= SpikeMoutput.num_spikes

    ans = np.zeros(2)
    ans[test_ansset[index]] = 1

    if spike_count[0] > spike_count[1] and ans[0] == 1:
        test_ans_correct += 1
    elif spike_count[0] < spike_count[1] and ans[1] == 1:
        test_ans_correct += 1
    
    delta_output = spike_count - ans
    # print('%d : ' %(i), abs(delta_output[0]))
    test_delta_sum += abs(delta_output[0])
    test_delta_cnt += 1

print('%d of %d samples are correct' %(test_ans_correct,test_epoch))
print('%d out of %d tries are dead' %(test_epoch-test_delta_cnt,test_epoch))
print('error rates are %f' %(test_delta_sum/test_delta_cnt))
'''
show_plot_for_test = False
if show_plot_for_test:
    print(SpikeMoutput.count)
    print(SpikeMoutput.num_spikes)

    plt.plot(StateMoutput.t/ms, StateMoutput.v[0], label = 'False')
    plt.plot(StateMoutput.t/ms, StateMoutput.v[1], label = 'True')
    plt.xlabel(index)
    plt.legend()
    plt.show()
'''
