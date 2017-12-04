import numpy as np
import sys
import json

json_path = '../../res/' + sys.argv[1] + '.json'
data_path = sys.argv[2]

with open(json_path, 'r') as f:
    json_file = json.load(f)
f.close()

with open(data_path, 'r') as f:
    data = [row.split(' ') for row in f.read().split('\n')[:-1]]
f.close()

leterDict = dict([('A', 0) , ('C', 1) , ('D', 2) , ('E', 3) , ('F', 4) , \
                  ('G', 5) , ('H', 6) , ('I', 7) , ('K', 8) , ('L', 9) , \
                  ('M', 10), ('N', 11), ('P', 12), ('Q', 13), ('R', 14), \
                  ('S', 15), ('T', 16), ('V', 17), ('W', 18), ('Y', 19), \
                  ('X', 21), ('*', 20)])

feature, ans = [], []
for i in data:
    tmp = []
    ans.append(int(i[0]))
    protein = i[1]
    position = i[3]
    for j in json_file[protein]['T'][2]:
        if str(j['position']) == position:
            for k in j:
                if k != 'position':
                    for a in k:
                        tmp.append(leterDict[a])
            break
    if len(tmp) != 33:
        print(j)
    feature.append(tmp)

print(np.array(feature).shape, np.array(ans).shape)
print(np.array(feature), np.array(ans))
np.save(data_path + '_X.npy', np.array(feature))
np.save(data_path + '_Y.npy', np.array(ans))

