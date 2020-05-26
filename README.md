# ntm - Neural Turing Machines in pytorch

A [Neural Turing Machines](https://arxiv.org/abs/1410.5401) implementation in pytorch.

The goal was to implement a simple NTM with 1 read head and 1 write head, to reproduce the original paper's results.


## Copy task

The copy task tests whether NTM can store and recall a long sequence of arbitrary information. The network is presented with an input sequence of random binary vectors followed by a delimiter flag. The target sequence is a copy of the input sequence. No inputs are presented to the model while it receives the targets, to ensure that there is no assistance.

The model is trained on sequences of 1 to 20 8-bit random vectors. In less than 50k iterations, the model usually becomes really accurate.

Here is the net output compared to the target for a sequence of 20.
![](images/copy_20.png)

Here is the net output compared to the target for a sequence of 100. Note that the network was only trained with sequences of 20 or less.
![](images/copy_100.png)

## Repeat copy task

As said in the paper, "the repeat copy task extends copy by requiring the network to output the copied sequence a specified number of times and then emit an end-of-sequence marker. [...]
The network receives random-length sequences of random binary vectors, followed by a scalar value indicating the desired number of copies, which appears on a separate input channel. To emit the end marker at the correct time the network must be both able to interpret the extra input and keep count of the number of copies it has performed so far. As with the copy task, no inputs are provided to the network after the initial sequence and repeat number.

The model is trained on sequences of 1 to 10 8-bit random vectors, with a repeat between 1 and 10.

Here is the model output for a sequence of 10 and a repeat of 10.
![](images/repeat_10_10.png)

Here it is for a sequence of 10 and a repeat of 20.
![](images/repeat_10_20.png)

Here it is for a sequence of 20 and a repeat of 10. Maybe it needs a bit more training here!
![](images/repeat_20_10.png)

## Usage

```bash
# installation
pip install -r requirements.txt
# to train
python copy_task.py --train
# to evaluate
python copy_task.py --eval

```

### References

1. https://github.com/loudinthecloud/pytorch-ntm/
2. https://github.com/MarkPKCollier/NeuralTuringMachine
