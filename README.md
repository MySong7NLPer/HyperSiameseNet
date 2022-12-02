# HyperSiameseNet
This repository provides the code of the paper [A Preliminary Exploration of Extractive Multi-Document Summarization in Hyperbolic Space](https://dl.acm.org/doi/10.1145/3511808.3557538) (CIKM2022).

:smile: [The new version of our paper](https://drive.google.com/file/d/1UhHRh8iAl-fZuODt5-qlSGZFXrfTyJs2/view?usp=share_link)(corrected some errors in the published version) 😉.

## Dependencies
- Python 3.7
- [PyTorch](https://github.com/pytorch/pytorch) 1.4.0
- [fastNLP](https://github.com/fastnlp/fastNLP) 0.5.0
- [pyrouge](https://github.com/bheinzerling/pyrouge) 0.1.3
	- You should fill your ROUGE path in metrics.py line 20 before running our code.
- [rouge](https://github.com/pltrdy/rouge) 1.0.0
	- Used in  the validation phase.
- [transformers](https://github.com/huggingface/transformers) 2.5.1

	
All code only supports running on Linux.


## CONTACT

For any question, feel free to create an issue, and we will try our best to solve. \
**If the problem is more urgent**, you can send an email to me at the same time (I check email almost everyday).

```
NAME: Mingyang Song
EMAIL: mingyang.song@bjtu.edu.cn
```




Our implementation is built on the source code from [MatchSum](https://github.com/maszhongming/MatchSum). Thanks for their work.



