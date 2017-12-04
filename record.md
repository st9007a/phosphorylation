# MusiteDeep Record

## Network Architecture

### CNN

| CNN | |
|---------------------------------|---------------------|
| Conv1D filter 1x1 channel 200   | regularization L1 0 |
| Dropout 0.75                    |                     |
| ReLU                            |                     |
| Conv1D filter 9x9 channel 150   | regularization L1 0 | 
| Dropout 0.75                    |                     |
| ReLU                            |                     |
| Conv1D filter 10x10 channel 200 | regularization L1 0 |
| ReLU                            |                     |

### Attention

| Seq |
|-----|
| unit 8 activation linear L1 2 |

| Fm |
|----|
| unit 10 activation linear L1 0.151948 |


### Merge

| Merge |
|---------------------------------|
| Dropout 0                       |
| Dense unit 149 activation relu  |
| Dropout 0.298224                |
| Dense unit 8 activation relu    |
| Dropout 0                       |
| Dense unit 2 activation softmax |

### Compile

- loss: binary_crossentropy
- optimizer: adam (default value)

### Train

- batch size: 1200 (?
- epochs: 500
