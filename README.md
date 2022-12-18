# Executive Summary
In this repository, we seek to compare the performance of multiple machine learning models in music genre classification by utilizing the popular GTZAN dataset, which contains 100 audio files of 30 seconds each - each pertaining to one of 10 music genres (classes). 

Music genre classification has a multitude of potential real-world use cases - specifically in music streaming sites and applications that utilize recommendation systems. We seek to determine the best models for this application. 

Let us visualize our data.

# Data Visualization
Here is a sample audio snippet from the dataset:
https://user-images.githubusercontent.com/78565736/208314633-c5968fc6-d5d0-4544-87e7-cb5e2bdac817.mp4

Here is that same audio file visualized as a sound wave:
![jazz_sound_wave](https://user-images.githubusercontent.com/78565736/208314679-496fe05a-49e1-462e-934a-f09604497b8d.png)

Here is that same sound wave converted to a Mel Spectrogram:
![jazz_melspectrogram](https://user-images.githubusercontent.com/78565736/208314770-b0526f7c-c769-42c9-b30a-57ce2307a2ab.png)

# Methodology and Notebook Explanations
We implemented, trained, and tested a variety of models in order to compare their performance. CNN_Models.ipynb contains pretrained ResNet18, ResNet50, and VGG16 models we finetuned to fit our data. Classifier_Models.ipynb contains various models/classifiers such as a decision tree classifier, random forest classifier, support vector machine, and more that we employed for our classification task. MLP_Models.ipynb contains a few perceptron models that we developed, tested, and sequentially built upon in order to increase test classification accuracy. In order to reduce redundancy, we only included tested models that are unique from one another as well as significant to our analysis, and omitted other redundant and insignifcant models that we analyzed
