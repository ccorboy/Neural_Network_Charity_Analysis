# Neural Network Charity Analysis
TensorFlow, Sci-Kit, Pandas, Python, Machine Learning

## Overview
This analysis will use machine learning and deep neural networks to help predict whether applicants will be successful if funded by AlphabetSoup, using a dataset containing more than 34,000 organizations that have received funding. The features provided in said dataset will help create a binary classifier capable of making this prediction. 

Neural networks are an advanced form of machine learning that aims to recognize features and patterns in input data. Once any patterns are recognized, a quantitative output is generated to privde a clear summary. Neural networks are modeled after the human brain in the form of a set of algorithms. They are made up of multiple layers: one input layer, 1 or more hidden layers, and an output layer. Each hidden layer contains multiple neurons that perform unique computations that are linked and weighed against each other until reaching the resulting output. The neural network model in this analysis will take in the dataset with organizations funded by Alphabet Soup as the input. We will use two hidden layers to sufficiently handle the amount of data in our CSV file. Finally, we will know whether or not applicants to Alphabet Soup will or will not be successful once receiving funding as our quantitative output layer. 

This project will be carried out in the following three steps:

- Preprocessing Data for a Neural Network Model
- Compile, Train, and Evaluate the Model
- Optimize the Model

## Results
### Data Preprocessing 
In order for our model to be able to learn effectively, the data we use must be preprocessed to implement usefull information. The machine learning model can only analyze quantitative data, therefore we must use `OneHotEncoder` to convert categorical variables into numerical data. Once all categorical variables are replaced with numerical data, we will create a new data frame made up of our old and new quantitative variables. Then we select our target and feature variables which will make up our training and testing data for the machine learning analysis. 

#### Target Variable
The target variable/dependent variable will be the `IS_SUCCESSFUL` column. We aim to build a model that predicts whether or not a company is successful after receiving funding from Alphabet soup. This column will either have a 1 for yes or 0 for no. We will separate `IS_SUCCESSFUL` from the dataframe and assign it to `y`.

#### Feature Variables
The feature variables are our input values, which is another way of saying our independent variables. They will influence our target value in the machine learning model. Our feature variables will be every other column of data in our new dataframe besides `IS_SUCCESSFUL`. The feature variables will be assigned to `X` in our training and testing split. 

#### Variables to Remove
The columns `EIN` and `NAME` will be dropped from the data frame because they will have no impact on the analysis. `EIN` is the identification number for each organization, and `NAME` is the name of each organization. These variables are unique to their index and will have no effect on the analysis. 

### Compiling, Training and Evaluating the Model
This part of the analysis will require us to determine the amount of inputs there will be before choosing how mamy hidden layers and how many neurons per layer we need to train and evaluate the model. Once the model is ready we can run it to determine the loss and accuracy of our model, whether our model is successful in determining our goal, and if we need to make any adjustments to the model is it fails. 

#### How many neurons, layers, and activation functions did you select for your neural network model, and why?
The neural network's input variable is set to equal the number of variables (34,000) in our dataframe. We will incorporate two hidden layers due to the size of our data. The second hidden layer will allow the model to evaluate interactions and identify complex relationships between weighted variables. The extra hidden layer will account for more learned information. The first hidden layer will have 80 neurons and the second hidden layer will have 30 neurons. Because we are using two hidden layers, we dont need neurons that are two-three times the size of our input layer. The activation functions for each hidden layer will be the Rectified Linear Unit (RELU) function beacause it will simplify data in a range from 0 to 1, making it easier for the model to process.  

#### Were you able to achieve the target model performance?
We fail in reaching the target model performance of 75%. The accuracy of our model is about 73% at most, with a loss of 56%. 

#### What steps did you take to try and increase model performance?
In an attempt to improve the model performance, first we add a hidden layer, then we use a different activation function, and after we increase our number of epochs. The idea behind adding a third hidden layer is to improve our model so that it accounts for more information. The third hidden layer with ten neurons does not improve the models performance, the loss is 56% and the accuracy is 73%. 

Using a different activation function fails to improve our models performance once again. The loss is again 56% and the accuaracy is 73%. Instead of `relu` we use `tanh` in this model. The `tanh` activation function increases the range of our weighted data from -1 to 1 and is mainly used to differentiate between two classes. 

Finally we try to increase the epochs from 100 to 150. This gives the neural network model more rounds to analyze the data. This could have decreased error in our model due to the increased optimization of training data. This method too fails to improve our model with the loss at 56% and the accuarcy at 73% once again. 

## Summary
The results of the model performance suggest our model is not fit to predict whether or not organizations will be successful in receiving funds from Alphabet Soup. The loss of the model is 56% which is extremely high. The lower the loss of the model, the better the model's predicitons will be. The accuarcy of the model did not reach the target model performance of 75%. The issue may lay in the size of our dataset; maybe adding more data would allow our model to perform better. When there is a lack of data it can lead to overfitting, which may have been an issue in the formation of the model. One way of avoding overfitting would be to use a Support Vector Machine (SVM) as the model.  The SVM model is more appropriate to use when analyzing binary classifiaction because it will focus on the bigger picture and it will avoid overfitting the data. SVMs pinpoint data points from two separate groups and can build models predict outcomes for linear and nonlinear data.  In straightforward binary classification problems, SVMs can be more reliable than deep learning models. 
