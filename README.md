# Phosphorylation

## Nameing Description

### onehot21

encode 20 kinds of amino acid and **pad** symbol to onehot vector.
**X** symbol encode to 0.05 for first 20 position, 0 for 21 position.

### onehot22

encode 20 kinds of amino acid, **pad**, **X** symbol to onehot vector.

### onehot21-nox

encode 20 kinds of amino acid, **pad** symbol to onehot vector.
ignore **X** symbol.

### onehot21-nopad

encode 20 kinds of amino acid, **X** symbol to onehot vector.
ignore **pad** symbol.

### nobn

disable all batch normalization

### validcnn

set padding = 'VALID' to all conv2d

### errorconv

for first convolution layer, use 1x21 size filter, 1x22 stride and **onehot22** encoding
