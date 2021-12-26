'''
This code is used to find the best validation epoch and to calculate the performance of the model.
How to run: 
$ python get_final_performance_numbers.py results/interaction_prediction_reddit.txt 

Paper: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. S. Kumar, X. Zhang, J. Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2019. 
'''

import sys,os
import numpy as np

fname = sys.argv[1]
network=os.path.basename(fname).split('_')[-1]


validation_performances = []
test_performances = []
val = []
test = []
f = open(fname, "r")
idx = -1
for l in f:
    if "Validation performance of epoch" in l:
        if val != []:
            validation_performances.append(val)
            test_performances.append(test)
        idx = int(l.strip().split("epoch ")[1].split()[0])
        val = [idx]
        test = [idx]
        
    if "Validation:" in l:
        val.append(float(l.strip().split(": ")[-1]))
    if "Test:" in l:
        test.append(float(l.strip().split(": ")[-1]))

if val != []:
    validation_performances.append(val)
    test_performances.append(test)

validation_performances = np.array(validation_performances,dtype=object)
#print(validation_performances)
test_performances = np.array(test_performances,dtype=object)


print('\n\n*** For file: %s ***' % fname)
best_val_idx = np.argmax(validation_performances[:,1])
print("best_val_idx:",best_val_idx)


if "interaction" in fname:
    metrics = ['Mean Reciprocal Rank', 'Recall@10']
    with open (os.path.dirname(fname)+'/'+'interaction_best_performance_%s'%network,'w') as f:
        f.write("Best validation epoch: %d\n" % best_val_idx)
        f.write('\n\n*** Best validation performance (epoch %d) ***\n' % best_val_idx)
        
        for i in range(len(metrics)):
            f.write(metrics[i] + ': ' + str(validation_performances[best_val_idx][i+1])+'\n')
    
    
        f.write('\n\n*** Final model performance on the test set, i.e., in epoch %d ***\n' % best_val_idx)
        for i in range(len(metrics)):
            f.write(metrics[i] + ': ' + str(test_performances[best_val_idx][i+1])+'\n')
    
    
    
    
    
    
    
else:
    metrics = ['AUC']
    with open (os.path.dirname(fname)+'/'+'state_change_best_performance_%s'%network,'w') as f:
        f.write("Best validation epoch: %d\n" % best_val_idx)
        f.write('\n\n*** Best validation performance (epoch %d) ***\n' % best_val_idx)
        
        for i in range(len(metrics)):
            f.write(metrics[i] + ': ' + str(validation_performances[best_val_idx][i+1])+'\n')
    
    
        f.write('\n\n*** Final model performance on the test set, i.e., in epoch %d ***\n' % best_val_idx)
        for i in range(len(metrics)):
            f.write(metrics[i] + ': ' + str(test_performances[best_val_idx][i+1])+'\n')





    
    
    
    
    
    




print("Best validation epoch: %d" % best_val_idx)
print('\n\n*** Best validation performance (epoch %d) ***' % best_val_idx)
for i in range(len(metrics)):
    print(metrics[i] + ': ' + str(validation_performances[best_val_idx][i+1]))
    
    


print('\n\n*** Final model performance on the test set, i.e., in epoch %d ***' % best_val_idx)
for i in range(len(metrics)):
    print(metrics[i] + ': ' + str(test_performances[best_val_idx][i+1]))
