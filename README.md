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

Our datasets consist of three files:
   *  **train_values.csv** - consists of 38 different features
   *  **train_labels.csv** - consists of corresponding label values
   *  **test_values.csv** - for making prediction on unseen data by our model
   
Attributes of our datasets along with their description are listed below:
  * **geo_level_1_id**, **geo_level_2_id**, **geo_level_3_id** (type: int): geographic
  region in which building exists, from largest (level 1) to most specific
  sub-region (level 3). Possible values: level 1: 0-30, level 2: 0-1427, level 3:
  0-12567.
  * **count_floors_pre_eq** (type: int): number of floors in the building before the
  earthquake.
  * **age** (type: int): age of the building in years.
  * **area_percentage** (type: int): normalised area of the building footprint.
  * **height_percentage** (type: int): normalised height of the building footprint.
  * **land_surface_condition** (type: categorical): surface condition of the land
  where the building was built. Possible values: n, o, t.
  * **foundation_type** (type: categorical): type of foundation used while building.
  Possible values: h, i, r, u, w.
  * **roof_type** (type: categorical): type of roof used while building. Possible
  values: n, q, x.
  * **ground_floor_type** (type: categorical): type of the ground floor. Possible
  values: f, m, v, x, z.
  * **other_floor_type** (type: categorical): type of construction used in higher than
  the ground floors** (except for the roof). Possible values: j, q, s, x.
  * **position** (type: categorical): position of the building. Possible values: j, o, s, t.
  * **plan_configuration (type: categorical): building plan configuration. Possible
  values: a, c, d, f, m, n, o, q, s, u.
  * has_superstructure_adobe_mud** (type: binary): flag variable that indicates if
  the superstructure was made of Adobe/Mud.
  * **has_superstructure_mud_mortar_stone** (type: binary): flag variable that
  indicates if the superstructure was made of Mud Mortar - Stone.
  * **has_superstructure_stone_flag** (type: binary): flag variable that indicates if
  the superstructure was made of Stone.
  * **has_superstructure_cement_mortar_stone** (type: binary): flag variable that
  indicates if the superstructure was made of Cement Mortar - Stone.
  * **has_superstructure_mud_mortar_brick** (type: binary): flag variable that
  indicates if the superstructure was made of Mud Mortar - Brick.
  * **has_superstructure_cement_mortar_brick** (type: binary): flag variable that
  indicates if the superstructure was made of Cement Mortar - Brick.
  * **has_superstructure_timber** (type: binary): flag variable that indicates if the
  superstructure was made of Timber.
  * **has_superstructure_bamboo** (type: binary): flag variable that indicates if the
  superstructure was made of Bamboo* .
  * **has_superstructure_rc_non_engineered** (type: binary): flag variable that
  indicates if the superstructure was made of non-engineered reinforced
  concrete .
  * **has_superstructure_rc_engineered** (type: binary): flag variable that
  indicates if the superstructure was made of engineered reinforced concrete.
  * **has_superstructure_other** (type: binary): flag variable that indicates if the
  superstructure was made of any other material.
  * **legal_ownership_status** (type: categorical): legal ownership status of the
  land where the building was built. Possible values: a, r, v, w.
  * **count_families** (type: int): number of families that live in the building.
  * **has_secondary_use** (type: binary): flag variable that indicates if the building
  was used for any secondary purpose.
  * **has_secondary_use_agriculture** (type: binary): flag variable that indicates if
  the building was used for agricultural purposes.
  * **has_secondary_use_hotel** (type: binary): flag variable that indicates if the
  building was used as a hotel.
  * **has_secondary_use_rental** (type: binary): flag variable that indicates if the
  building was used for rental purposes.
  * **has_secondary_use_institution** (type: binary): flag variable that indicates if
  the building was used as a location of any institution.
  * **has_secondary_use_school** (type: binary): flag variable that indicates if the
  building was used as a school.
  * **has_secondary_use_industry** (type: binary): flag variable that indicates if
  the building was used for industrial purposes.
  * **has_secondary_use_health_post** (type: binary): flag variable that indicates
  if the building was used as a health post.
  * **has_secondary_use_gov_office** (type: binary): flag variable that indicates if
  the building was used as a government office.
  * **has_secondary_use_use_police** (type: binary): flag variable that indicates if
  the building was used as a police station.
  * **has_secondary_use_other** (type: binary): flag variable that indicates if the
  building was secondarily used for other purposes.
  
  We are going to predict _damage_grade_ class, which represents a level of damage
  to the building that was hit by the earthquake. There are 3 grades/classes of the
  damage:
 * **1** represents low damage
 * **2** represents a medium amount of damage
 * **3** represents almost complete destruction
  
## 3. Exploratory Data Analysis (EDA) and Visualization

It is a valuable tool for understanding and gaining insights from data, and uncovering any issues or anomalies. It can also be used to generate ideas for further research or to communicate findings to others, and is an important step in the machine learning process.

Some common EDA steps we followed in the project are:

* __Overview of the data:__  To provide a summary of the data, including the number of rows and columns, column names, data types, and missing values.

* __Summarizing the data:__ To get summary statistics for the numerical features in the dataset, such as count, mean, standard deviation, minimum, and maximum values to get a sense of the central tendency and spread of the data. We also performed summary statistics for the categorical features in the dataset, such as count, unique values, and the most common value.

* __Checking for missing values:__ To make sure there are no missing values in the data set, as these can cause issues with analysis and modeling.

* __Distribution of each binary feature:__ To get the proportion of each binary value to understand the distribution of binary features.

* __Visualize the distribution of the target variable (damage_grade):__ To understand the balance of the classes in the dataset.

* __Visualize the distribution of each numeric feature and detect outliers:__ To identify any unusual values or patterns that could be causing skews in the data.

* __Visualize the correlation between the numeric features:__ To identify any strong correlations or multicollinearity between the features.

## 4. Data Preprocessing and Handling Imbalanced Datasets

Data preprocessing is a technique that is used to convert raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for analysis. Therefore, certain steps are executed to convert the data into a small clean data set.

Some common data preprocessing steps we followed are:

* __Winsorization:__ To handle the outliers in each numerical feature and get rid of the effect of extreme values. Specifically, for each continuous feature, we applied winsorization with limits=[0.05, 0.05] using the winsorize() function from the scipy.stats library.

* __Standard Feature scaling:__ To normalize the numerical features to have mean=0 and standard deviation=1.

* __Encoding of categorical features:__ To convert the categorical features to numerical features. Specifically, we used the get_dummies() function from the pandas library to perform one-hot encoding on the specified columns.

* __Variance thresholding:__ To remove low variance features that may not contribute much to the model's predictive power. Specifically, we used the VarianceThreshold() function from the sklearn.feature_selection library with a threshold of 0.02 to remove low variance features.

* __Handling the imbalanced dataset:__ Apply different techniques such as Random Undersampling, Random Oversampling, and SMOTE. Random Undersampling was used to reduce the number of samples in the majority class. Random Oversampling was used to increase the number of samples in the minority class. Finally, we used SMOTE (Synthetic Minority Over-sampling Technique) to generate synthetic samples for the minority class. We applied these techniques using the RandomUnderSampler(), RandomOverSampler(), and SMOTE() functions from the imblearn library.

## 5. Model Selection and Fitting
Model selection and fitting is a critical step in the process of creating a mathematical or statistical model that accurately represents the relationships and patterns in a given dataset. This step is a common task in machine learning, as it allows you to make predictions or inferences about the data based on the patterns identified in the model.

In our project, we applied various machine learning algorithms to predict the damage grade of buildings. The algorithms used for model training and evaluation included:
 * Random Forest Classifier
 * Decision Tree Classifier
 * XGBoost 
 * KMeansClassifier
 * Softmax Regression
 
These models were trained on our dataset to predict the level of damage to buildings caused by earthquakes.

## 8. Performance evaluation
It is an important step in the model building process, as it allows you to assess the effectiveness of the model and make any necessary adjustments to improve its performance. It is also important to evaluate the performance of a model on unseen data, as this can provide a more realistic assessment of its performance on real-world tasks. 

To calculate evaluation metrics we performed the following steps:

* Split the data into a training set and a test set.
* Fit the model to the training set.
* Use the model to make predictions on the test set.
* Calculate the evaluation metrics using the predictions and the true values.

We evaluated the performance of each model using the *micro-averaged F1-score*, a widely used performance metric in machine learning. The micro-averaged F1 score is a metric that makes sense for multi-class data distributions. It is a suitable performance metric for imbalanced datasets because it takes into account both precision and recall of the minority class.

<u>For example<u>:  In the case of our earthquake damage prediction dataset, the majority class (level 2 damage) has significantly more instances than the other two classes (level 1 and 3 damage), making it an imbalanced dataset. In such cases, accuracy can be misleading because it may be high due to the high number of correctly classified majority class instances, while the minority class instances are misclassified. F1-score considers both precision and recall, and micro-averaging of F1-score provides an overall score that takes into account the performance of all classes.

## 9. Making Predictions
After having trained a machine learning model, you can use it to make predictions on our own data.

<br>
