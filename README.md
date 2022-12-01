# ProjectPaperOversampling
An implementation of . 
This code was built based on [Yannael Leborgne generator](https://github.com/).

# Dependencies

To run this code fully, you'll need [this](https://pytorch.org/) (we're using version 1.4.0), [that](https://scikit-learn.org/stable/).
We've been running our code in Python 3.7.


## Deep active learning + Semi-supervised learning

|                Sampling Strategies                |    Year    | Done |
|:-------------------------------------------------:|:----------:|:----:|
|               Consistency-SSLAL [16]                |  ECCV'20  |  ✅ |
|               MixMatch-SSLAL [17]                |  arXiv  |  ✅ |
|               UDA [18]                |  NIPS'20  |  In progress |




# Running an experiment
## Requirements

First, please make sure you have installed Conda. Then, our environment can be installed by:
```
conda create -n ProjectPaperOversampling python=3.7
conda activate ProjectPaperOversampling
pip install -r requirements.txt
```

## Example
```
python main.py --param1 value1 --param2 value2
```
It runs .... using ... and ... data. The result will be saved in the **./save** directory.

You can also use `run.sh` to run experiments.


# Contact
If you have any questions/suggestions, or would like to contribute to this repo, please feel free to contact:
  Daniele Lunghi `dlunghi@ulb.ac.be`,   Gian Marco Paldino `gpaldino@ulb.ac.be`

  
## References

[1] (Author, Conference'22) Title [paper](https://arxiv.org/pdf/) [code](https://github.com/)


