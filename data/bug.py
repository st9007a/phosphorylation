with open('validation_data', 'r') as f:
    data = f.read().split('\n')[:-1]
f.close()
answer = []
for i in data:
    i = i.split(' ')
    if i[0] == '1':
        answer.append(1)
    else:
        answer.append(0)
import numpy as np
print(answer)
np.save('validation_data_Y.npy', np.array(answer))
print(np.array(answer).shape)
