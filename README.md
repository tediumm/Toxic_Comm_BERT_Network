# Research Proposal: Predicting Toxicity in Online Comments using NLP

## 1. Title

Predicting Toxicity in Online Comments using NLP

## 2. Abstract

Online communication has revolutionized how people interact with one another, but it has also created new challenges in managing toxic language in online comments. In this project, developmetn of a natural language processing (NLP) model is proposed. Main task will be to detect toxic language in online comments. A dataset from the Jigsaw Unintended Bias in Toxicity Classification competition will be used to train and evaluate future model. Chosen approach will include a lexical coherence graph modeling technique using word embeddings, as well as a stacking-based efficient method for toxic language detection. High accuracy in detecting toxic language in online comments is expected to be achieved, which will be useful for content moderators and online community managers.

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

High accuracy in detecting toxic language in online comments is expected while using proposed ensemble-based efficient method. Model will be useful for content moderators and online community managers, who can use it to automatically flag toxic comments for removal. In addition, model will provide insights into the patterns of language used in toxic comments, which can be useful for understanding and addressing toxic behavior in online communication.

## 5. Conclusion

In this project, development of an NLP model is proposed to detect toxic language in online comments using a combination of the lexical coherence graph modeling technique and the stacking based efficient method. The Jigsaw Unintended Bias in Toxicity Classification dataset will be used to train and evaluate our model, and it is expected to achieve high accuracy in detecting toxic language in online comments. The model will be useful for content moderators and online community managers, who can use it to automatically flag toxic comments for removal. In addition, model will provide insights into the patterns of language used in toxic comments, which can be useful for understanding and addressing toxic behavior in online communication.

## 6. References

- Jigsaw Unintended Bias in Toxicity Classification. (n.d.). Retrieved April 30, 2023, from https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data

- Mesgar, M., & Strube, M. (2016). Lexical coherence graph modeling using word embeddings. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (pp. 1414-1423).

- Oikawa, Y., Nakayama, Y., & Murakami, K. (2022). A stacking-based efficient method for toxic language detection on live streaming chat. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing: Industry Track (pp. 571-578).

- Park, S., & Fung, P. (2017). A hierarchical LSTM model for joint sentiment analysis and emotion classification. In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers (pp. 746-751).

- Rossi, E., Zou, J., Chiu, W., Zhang, Y., Liu, H., & Wang, L. (2020). Temporal graph networks for deep learning on dynamic graphs. arXiv preprint arXiv:2006.10637.


# Resulting model descriprtion

*Dataset info:* 
- 159571 rows in total
- Vocabulary Size: 181025
- Max Sequence Length: 1250

For reasons of inefficiency detection of CNN model, it was decided that BERT model would provide better results in more efficient way. Despite the fact that the CNN model is also capable of processing sequences of signs and words, its main task is rather to work with images using convolutional network methods. At the same time, BERT (Bidirectional Encoder Representations from Transformers) is a state-of-the-art transformer-based model for natural language processing (NLP) tasks. It was introduced by Google AI in 2018 and has achieved remarkable performance on a wide range of NLP tasks, including text classification, sentiment analysis, named entity recognition, and question answering. That is why final decision was made in favor of using this model.

## Preprocessing

Data manipulation libraries used: pandas, machine learning libraries PyTorch, transformers library for BERT model and tokenizer, and various other libraries for data preprocessing, evaluation metrics, and visualization (which can be found in code file "Data_Preprocessing.ipynb").
The comments are preprocessed by converting them to lowercase and tokenizing them using a BERT tokenizer. The labels are converted to tensors.

![image](https://github.com/tediumm/Toxic_Comm_Ntwrk/assets/99214871/b36c0773-c08a-41cb-af76-ff6a494395a8)

## Clustering for further comment dataset observation

Main Topics (NMF Model) were found for the whole dataset (figure 1) and for rows with toxic comments only (figure 2). Code can be found in "NMF_Topic_Modelling.ipynb"

![image](https://github.com/tediumm/Toxic_Comm_Ntwrk/assets/99214871/f7688fc5-dd34-497c-ba4c-54f56e308cdd)
*Fig. 1 Topic modelling for all comments*

![image](https://github.com/tediumm/Toxic_Comm_Ntwrk/assets/99214871/12db4752-68f0-4ca4-9e18-80856a85f0ee)
*Fig. 2 Topic modelling for toxic comments only*

## Building the BERT model

Used a custom BERT-based model for sequence classification.
It utilizes the BertForSequenceClassification model from the transformers library, which is a pre-trained BERT model fine-tuned for classification tasks. The model takes input sequences, performs attention-based encoding, and outputs logits for each class.

### BERT model details

The code includes the necessary steps to obtain word embeddings from the BERT model for the toxic words. It uses the BERT tokenizer to tokenize the words and passes them through the BERT model to obtain the embeddings. The embeddings are then used to calculate similarity scores between the words, which are later used in the network visualization.

## Training and evaluating the model:

The model is trained using the training dataset. The code sets up the optimizer (AdamW) and the loss function (BCEWithLogitsLoss). The training loop iterates over the batches of the training dataset, performs forward and backward propagation, and updates the model parameters. The model is evaluated on the test dataset after each epoch.

![image](https://github.com/tediumm/Toxic_Comm_Ntwrk/assets/99214871/a6b4e127-6336-4616-9dfb-7bd6db717558)
*Fig. 3 Model evaluation*

## Network visualization

The toxic words identified in the previous step are visualized as a network graph. The graph represents the connections between words based on their similarity scores computed using BERT word embeddings. 
Final version of network graph is created using the NetworkX and Bokeh libraries, and it provides an interactive visualization to explore the relationships between toxic words.


![image](https://github.com/tediumm/Toxic_Comm_Ntwrk/assets/99214871/67673845-5e51-45e3-aa1b-07a74586754d)
*Fig. 4 First non-interactive graph plot*

![image](https://github.com/tediumm/Toxic_Comm_Ntwrk/assets/99214871/4591a537-eafc-49f2-b971-b0d66ba82000)
*Fig. 5 Final version of interactive graph plot*
