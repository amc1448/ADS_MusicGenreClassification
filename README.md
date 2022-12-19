# Executive Summary
In this repository, we compared the performance of multiple machine learning models in music genre classification by utilizing the popular GTZAN dataset, which contains 1000 audio files of 30 seconds each - each pertaining to one of 10 music genres (classes). The genres in the dataset include blues, classical, country, disco, hihop, jazz, metal, pop, reggae, and rock.

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
We implemented, trained, and tested a variety of models in order to compare their performance. CNN_Models.ipynb contains pretrained ResNet18, ResNet50, and VGG16 models we finetuned to fit our data. Classifier_Models.ipynb contains various models/classifiers from the SKLearn package including a decision tree classifier, random forest classifier, support vector machine, and more that we employed for our classification task. MLP_Models.ipynb contains a few perceptron models that we developed, tested, and sequentially built upon in order to increase test classification accuracy. During the development process, we evaluated additional models within each of the three categories—but ultimately only included the models that proved more significant to our analysis and were not redundant to other models included.

# Results
## SKLearn Classifiers
Among the SKLearn classifiers, we found that the Random Forest Classifier and the KNN Classifier performed the best on the dataset, with accuracies of 0.81415 and 0.80581, respectively. Both models had high accuracy across the board, but generally made mistakes in the same places—confusing many of the same genres most: disco and rock; rock and country; and jazz and classical.

| Classifier    | Accuracy      |
| ------------- | ------------- |
| Random Forest | 0.814         |
| KNN           | 0.805         |
| SVC           | 0.754         |
| Logistic Reg  | 0.698         |
| SGD           | 0.655         |
| Decision Tree | 0.646         |
| Adabooster    | 0.452         |

## Interpretation and Explanation of Our Results
The Random Forest Classifier performed best of all classifiers listed above. This is likely due to the features having a low correlation to each other, and thus allowing for more distinct decisions to be made for each tree in the forest. In addition, a number of estimators of 1000 proved to be sufficiently high for good accuracy results. Notice that the Decision Tree classifier performed relatively well by itself. This is an indicator that the Random Forest Classifier would perform even better.

The K-Neighbors Classifier also performed quite well on the dataset. We used a K-value of 19 because it is where we saw accuracy peak. K-values of less than and more than 19 appeared to have a lower accuracy, so 19 is likely an optimal value. KNN is a common algorithm used in recommendation services, so it is logical that the K-Neighbors Classifier would perform well in a music genre classication role.

The Support Vector Classifier performed relatively highly, so 

## Multi-Layer Perceptrons
We also tested two simple Multi-Layer Perceptron models. We built the models using Tensorflow. The first model consisted of just four dense layers, while the second added dropout layers and a fifth dropout layer. Both models were trained with an adamn optimizer and for 100 epochs—though accuracy for both models largely plateauted after 20 epochs. 

| Classifier      | Accuracy      |
| --------------- | ------------- |
| 4 DLs           | 0.902         |
| 5 DLs + Dropout | 0.913         |

Model 1

![image](https://user-images.githubusercontent.com/98373786/208317549-06da95d5-ebb2-4a60-9009-c851a7991cb1.png)

Model 2

![image](https://user-images.githubusercontent.com/98373786/208317585-ab0c1df1-1575-47e3-9a69-038ef55cfbbc.png)

Both models performed well on the dataset, and outperformed all of the other kinds of models tested across both the SKLearn classifiers and the CNNs. The model that included dropout layers slightly outperformed the model that did not, but the difference between them was not as significant as we had initially expected.

## CNNs
To use the pre-trained CNN models, we used the mel spectrogram image representations of the sound files in the dataset instead of working from the features, like we did for the SKLearn classifiers and the Multi-Layer Perceptron models. The three models we chose to try are VGG16, ResNet18, and ResNet50. First, we processed the data and loaded it into a dataloader using PyTorch. After calling the models, each model was trained for 100 epochs. The two ResNet models were initialized with an adam optimizer, a learning rate of 0.001, and Cross Entropy Loss. The VGG model used an SGD optimizer and a learning rate of 0.01.

| Classifier    | Accuracy      |
| ------------- | ------------- |
| ResNet18      | 0.717         |
| ResNet50      | 0.677         |
| VGG16         | 0.535         |

Of the three models, ResNet18 performed best with an accuracy of 71.7%, despite being the least complex and layered. The two more complex networks' lesser performance is reflective of overfitting to the training data, as well as the vanishing gradient problem: that as CNNs become deeper, the networks' ability to backpropagate useful gradient information to initial layers is weakened—weakening the respective model’s performance on a whole.



