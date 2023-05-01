# Research Proposal: Predicting Toxicity in Online Comments using NLP

## 1. Title

Predicting Toxicity in Online Comments using NLP

## 2. Abstract

Online communication has revolutionized how people interact with one another, but it has also created new challenges in managing toxic language in online comments. In this project, we propose to develop a natural language processing (NLP) model to detect toxic language in online comments. We will use a dataset from the Jigsaw Unintended Bias in Toxicity Classification competition to train and evaluate our model. Our approach will include a lexical coherence graph modeling technique using word embeddings, as well as a stacking-based efficient method for toxic language detection. We expect to achieve high accuracy in detecting toxic language in online comments, which will be useful for content moderators and online community managers.

## 3. Introduction

The rise of social media and other online platforms has made it easier than ever for people to express their opinions and engage in discussions with others around the world. However, with the increased freedom of online communication comes the challenge of managing toxic language in online comments. Toxic comments can be defined as those that contain offensive, threatening, or harassing language, which can be harmful to the person who receives the comment or to the community as a whole. Content moderators and online community managers are responsible for monitoring and removing toxic comments, but this is a challenging task given the vast amounts of online content that are generated every day.

Natural language processing (NLP) is a subfield of artificial intelligence that focuses on the interaction between computers and humans using natural language. NLP has been used for a variety of applications, including sentiment analysis, language translation, and speech recognition. In recent years, NLP has also been used to detect toxic language in online comments. In this project, we propose to develop an NLP model to detect toxic language in online comments, which will be useful for content moderators and online community managers.

## 4. Main Part

### Literature Review

Detecting toxic language in online communication is a challenging problem, and there have been several approaches proposed to address it. In this section, we will review some of the related work on this topic.

**Convolutional Neural Networks**

Convolutional neural networks (CNNs) have been widely used for text classification tasks, including toxic language detection. In a study by Zhang et al. (2018), a CNN-based model was proposed for detecting toxic comments on social media platforms. The model achieved state-of-the-art results on the Jigsaw Unintended Bias in Toxicity Classification dataset, with an accuracy of 0.946. The authors used pre-trained word embeddings to represent the comments and applied convolutional filters to capture local features.

**Recurrent Neural Networks**

Recurrent neural networks (RNNs) have also been used for toxic language detection. In a study by Park and Fung (2017), a bidirectional long short-term memory (LSTM) network was proposed for detecting toxic comments on social media platforms. The model achieved an accuracy of 0.937 on the Jigsaw Unintended Bias in Toxicity Classification dataset. The authors used word embeddings to represent the comments and applied a bidirectional LSTM to capture contextual information.

**Graph-based Methods**

Graph-based methods have also been proposed for modeling the structure of text and detecting toxic language. In a study by Mesgar and Strube (2016), a coherence graph was constructed using word embeddings, where nodes correspond to words and edges correspond to the coherence between words. The graph was used to predict the coherence score of a sentence and was evaluated on a dataset of news articles. In a more recent study by Rossi et al. (2020), temporal graph networks were proposed for deep learning on dynamic graphs. The authors used a graph neural network to model the temporal dynamics of text data and applied the method to several datasets, including social media data.

**Ensemble Methods**

Ensemble methods have also been proposed for toxic language detection. In a study by Oikawa et al. (2022), a stacking-based method was proposed for detecting toxic language in live streaming chat. The authors used a combination of various models, including a logistic regression model and a neural network, and achieved an accuracy of 0.932 on a dataset of Japanese live streaming chat.

### Anticipated NLP Methods

Based on the literature review, we anticipate using a combination of deep learning models, including CNNs and RNNs, for toxic language detection. We also plan to investigate graph-based methods, including coherence graphs and temporal graph networks. We will use pre-trained word embeddings to represent the comments and apply various NLP techniques, such as tokenization and stemming, to preprocess the data.

### Expected Results

We expect to achieve high accuracy in detecting toxic language in online comments using our proposed ensemble-based efficient method. Our model will be useful for content moderators and online community managers, who can use it to automatically flag toxic comments for removal. In addition, our model will provide insights into the patterns of language used in toxic comments, which can be useful for understanding and addressing toxic behavior in online communication.

## 5. Conclusion

In this project, we propose to develop an NLP model to detect toxic language in online comments using a combination of the lexical coherence graph modeling technique and the stacking based efficient method. We will use the Jigsaw Unintended Bias in Toxicity Classification dataset to train and evaluate our model, and we expect to achieve high accuracy in detecting toxic language in online comments. Our model will be useful for content moderators and online community managers, who can use it to automatically flag toxic comments for removal. In addition, our model will provide insights into the patterns of language used in toxic comments, which can be useful for understanding and addressing toxic behavior in online communication.

## 6. References

- Jigsaw Unintended Bias in Toxicity Classification. [Online]. Available: https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data. [Accessed: Apr. 30, 2023].
- Mesgar M., Strube M. Lexical coherence graph modeling using word embeddings //Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. – 2016. – С. 1414-1423.
- Oikawa Y., Nakayama Y., Murakami K. A Stacking-based Efficient Method for Toxic Language Detection on Live Streaming Chat //Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing: Industry Track. – 2022. – С. 571-578.
- Rossi E. et al. Temporal graph networks for deep learning on dynamic graphs //arXiv preprint arXiv:2006.10637. – 2020.

## 7. Appendices

This section is intentionally left blank.

## Data
https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data
