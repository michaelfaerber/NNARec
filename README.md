# NNARec: Finding the Right Neural Network Architecture Given a Research Problem

## Abstract
Considering the increasing rate of scientific papers published in recent years, for researchers throughout all disciplines it became a challenge to keep track of which latest scientific methods are suitable for which applications. In particular, in the past, various neural network architectures were published. In this work, we propose the task of '''recommending neural network architectures based on textual problem descriptions'''. We frame the recommendation as a text classification task and develop appropriate text classification models for this task. In experiments based on three data sets, we find that an SVM classifier outperforms a more complex model based on BERT. Overall, we give evidence that neural network architecture recommendation is a nontrivial but gainful research topic. 

## Contributions

1. We create evaluation data sets for neural network architecture recommendation, consisting of 66 unique architectures and 284,337 textual problem descriptions. 
2. We train and evaluate several classifiers capable of predicting neural network architectures based on textual problem descriptions.

More information can be found in our paper.

## Contact
The system has been designed and implemented by Michael Färber and Nicolas Weber. Feel free to reach out to us:

[Michael Färber](https://sites.google.com/view/michaelfaerber), michael.faerber@kit&#46;edu

## How to Cite
Please cite our work as follows:
```
@inproceedings{Faerber2022SDU,
  author    = {Michael F{\"{a}}rber and
               Nicolas Weber},
  title     = "{Finding the Right Neural Network Architecture Given a Research Problem}",
  booktitle = "{Proceedings of the AAAI-22 Workshop on Scientific Document Understanding}",
  series    = "{SDU@AAAI'22}",
  year      = {2022}
}
```
