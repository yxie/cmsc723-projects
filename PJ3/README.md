## environment setup for windows  
```
conda create -n pytorch python=3.6 scipy numpy matplotlib nltk
# this is a custom build pytorch. version 0.2.1
conda install -n pytorch -c peterjc123 pytorch
```
you may need to modify source code at torch/backends/cudnn/__ init__.py  
add this code to line number 20  
```
__cudnn_version = lib.cudnnGetVersion()
```  


## install pytorch 0.3 for windows  
If you find a need to upgrade to 0.3 use this. (0.3 have some additional features such as variable.shape())  
refer to [this link](https://github.com/peterjc123/pytorch-scripts)  
download appropriate whl package from [here](https://ci.appveyor.com/project/peterjc123/pytorch-elheu/branch/v0.3.0_win/job/sa40xceop5g9jsee/artifacts)  
```bash
conda create -n pytorch_3 python=3.6 numpy matplotlib nltk MKL PyYAML  
activate pytorch_3
cd /to/whl/folder
pip install <name of package>.whl
```



## discuss  
1. possible improve:  
    1. bi-directional GRU
    2. peephole(LSTM only): allow gates depends not only on previous hidden state but also on previous memory/internal station
    3. stack of GRU: multiple layers. promote non-linearity. generally improve accuracy. more layers are unlikely to make a big difference and may lead to overfitting.
    4. Truncated BPTT: only backprop constant steps. Help with runtime.
    5. batch update:  
    6. regularization: dropout, L1, L2, etc.  
    7. multiple epochs and watch for validation accuracy. 
    8. tune learning rate.
    9. other optimizer: Adagrad, Adam, etc.
    * links that have some suggestion:  
    [A Beginner’s Guide to Recurrent Networks and LSTMs](https://deeplearning4j.org/lstm.html)  

2. problem:
    1. overfitting: size of parameter vs. number of training sample  
    
## resources
* basics:  
[lecture slide seq2seq](http://www.cs.umd.edu/class/fall2017/cmsc723/slides/slides_16.pdf)  
[lecture slide attention](http://www.cs.umd.edu/class/fall2017/cmsc723/slides/slides_17.pdf)  
[lecture slide LSTM](http://mt-class.org/jhu/slides/lecture-nn-lm.pdf)  
[understanding LSTM](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)  
[Implementing a GRU/LSTM](http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/)  
[pytorch GRU doc](http://pytorch.org/docs/master/nn.html)  




* improve:






