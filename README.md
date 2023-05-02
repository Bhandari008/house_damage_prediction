## House Damage Prediction Using Machine Learning



**Note:**
Create folders inside notebook/data with foldername as SMOTE, Original, OverSampling, UnderSampling



MlFlow Results:
---

Multinomial Logistic Regression
![image](https://user-images.githubusercontent.com/103937888/235445895-e978e3a5-f829-40f3-98cc-5e6f8eca0f0e.png)

 # Methodology
The project task is accomplished under the following steps:

## 1. Importing necessary libraries
Importing library in a Python script allows you to use the functions, classes, and other objects defined in those libraries in your code and makes it easier to accomplish tasks.

## 2. Load Dataset
Loading a dataset is an important step in the machine learning process because it allows you to access the data and begin working with it. There are many different ways to load a dataset, depending on where the data is stored and how it is formatted.

## 3. Exploratory Data Analysis (EDA)
It is a valuable tool for understanding and gaining insights from data, and uncovering any issues or anomalies. It can also be used to generate ideas for further research or to communicate findings to others.and is an important step in the data science process.

Some common techniques we used in EDA include:

* __Visualizing the data:__ Plotting the data help you get a sense of the distribution and relationships between variables. for eg: I plotted the dataset in the scatter plot and find the line (regression line) of best fit.

* __Summarizing the data:__ Calculating summary statistics such as mean, median, and standard deviation can help you get a sense of the central tendency and spread of the data.

* __Checking for missing values:__ Make sure there are no missing values in the data set, as these can cause issues with analysis and modeling.

* __Checking for outliers:__ Look for any unusual or extreme values that could be causing skews in the data.

## 4. Model building
It is the process of creating a mathematical or statistical model to represent the relationships and patterns in a dataset. Model building is a common task in data science and machine learning, as it allows you to make predictions or inferences about the data based on the patterns identified in the model.

There are many different types of models that can be built, including linear models, logistic regression models, decision trees, and neural networks. The choice of model will depend on the type of data and the specific goals of the analysis.

I used `Logistic Regression` model in this project.

## 8. Performance evaluation
It is an important step in the model building process, as it allows you to assess the effectiveness of the model and make any necessary adjustments to improve its performance. It is also important to evaluate the performance of a model on unseen data, as this can provide a more realistic assessment of its performance on real-world tasks. 

To calculate evaluation metrics for a linear regression model, I performed use the following steps:

* Split the data into a training set and a test set.
* Fit the model to the training set.
* Use the model to make predictions on the test set.
* Calculate the evaluation metrics using the predictions and the true values.

Some common *performance evaluation metrics* for Linear Regression are:
* __Mean Absolute Error (MAE):__ It is a measure of how close the predictions are to the true values, with a lower MAE indicating a better fit.
*  __Mean Squared Error (MSE):__ It is a measure of the average squared difference between the predictions and the true values, with a lower MSE indicating a better fit.
* __R^2 score:__ It is a measure of the goodness of fit of the model, with a higher R^2 score indicating a better fit.

## 9. Making Predictions
After having trained a machine learning model, you can use it to make predictions on our own data. I did this by calling a prediction function on the model, passing in the new data as an input.

For example: The model predicted score above 90% if a student studies for 9.25 hrs/ day.

<br>

__Thank You and Happy Learning!__
