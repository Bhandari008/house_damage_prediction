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
  
## 3. Exploratory Data Analysis (EDA)

It is a valuable tool for understanding and gaining insights from data, and uncovering any issues or anomalies. It can also be used to generate ideas for further research or to communicate findings to others, and is an important step in the machine learning process.

Some common techniques we used in EDA include:

* __Visualizing the data:__ Plotting the data help you get a sense of the distribution and relationships between variables. for eg: I plotted the dataset in the scatter plot and find the line (regression line) of best fit.

* __Summarizing the data:__ Calculating summary statistics such as mean, median, and standard deviation can help you get a sense of the central tendency and spread of the data.

* __Checking for missing values:__ Make sure there are no missing values in the data set, as these can cause issues with analysis and modeling.

* __Checking for outliers:__ Look for any unusual or extreme values that could be causing skews in the data.

## 4. Model building
It is the process of creating a mathematical or statistical model to represent the relationships and patterns in a dataset. Model building is a common task in machine learning and machine learning, as it allows you to make predictions or inferences about the data based on the patterns identified in the model.

There are many different types of models that can be built, including linear models, logistic regression models, decision trees, and neural networks. The choice of model will depend on the type of data and the specific goals of the analysis.

We used `Logistic Regression` model in this project.

## 8. Performance evaluation
It is an important step in the model building process, as it allows you to assess the effectiveness of the model and make any necessary adjustments to improve its performance. It is also important to evaluate the performance of a model on unseen data, as this can provide a more realistic assessment of its performance on real-world tasks. 

To calculate evaluation metrics we performed the following steps:

* Split the data into a training set and a test set.
* Fit the model to the training set.
* Use the model to make predictions on the test set.
* Calculate the evaluation metrics using the predictions and the true values.

The *performance evaluation metrics* we used for analysis in our project is *micro-averaged F1-score*. The micro-averaged F1 score is a metric that makes sense for multi-class data distributions. It is a suitable performance metric for imbalanced datasets because it takes into account both precision and recall of the minority class. 

<u>For example<u>:  In the case of our earthquake damage prediction dataset, the majority class (level 2 damage) has significantly more instances than the other two classes (level 1 and 3 damage), making it an imbalanced dataset. In such cases, accuracy can be misleading because it may be high due to the high number of correctly classified majority class instances, while the minority class instances are misclassified. F1-score considers both precision and recall, and micro-averaging of F1-score provides an overall score that takes into account the performance of all classes.

## 9. Making Predictions
After having trained a machine learning model, you can use it to make predictions on our own data.

<br>
