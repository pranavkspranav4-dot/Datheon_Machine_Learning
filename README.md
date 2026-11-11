Machine Learning & Data Science Internship Projects
This repository contains three Jupyter Notebook projects completed as
part of machine learning and data science practice/internship tasks. The
notebooks cover exploratory data analysis, data preprocessing,
classification, regression, model evaluation, and basic NLP.
Projects
1. Iris Flower Classification
Notebook:
Datheon_Machine_Learning_&_Data_Science_Internship_TASK1.ipynb

Objective: Classify Iris flowers into their species using sepal and
petal measurements.

Dataset: iris.csv
The notebook works with 150 records and four input features: - Sepal
length - Sepal width - Petal length - Petal width

Workflow: 1. Load and inspect the dataset using Pandas. 2. Check
descriptive statistics and missing values. 3. Explore the data using
Seaborn visualizations (bar plot, count plot, pair plot, scatter plot,
histogram, and correlation heatmap). 4. Encode the target species using
LabelEncoder. 5. Split the data into training and testing sets
(80/20). 6. Scale the features using StandardScaler. 7. Train Logistic
Regression and Decision Tree Classifier models. 8. Evaluate the models
using accuracy, precision, recall, F1-score, confusion matrix, and
classification report.

Results recorded in the notebook: - Logistic Regression test
accuracy: 1.00 (100%) - Decision Tree test accuracy: 1.00 (100%)

These scores are from the notebook's particular train/test split
(random_state=42). They should not be interpreted as a guarantee of
the same performance on unseen datasets.
2. House Price Prediction
Notebook: TASK2_HOUSING.ipynb

Objective: Predict median house value using housing-related
features.

Dataset: Housing.csv
The notebook contains 20,640 rows and 10 columns, including location,
housing age, rooms, bedrooms, population, households, median income,
ocean proximity, and median house value.

Workflow: 1. Load and inspect the dataset. 2. Check missing values
and fill missing total_bedrooms values with the column mean (207
missing values were reported). 3. Explore numeric feature correlations
using a heatmap. 4. Encode ocean_proximity using LabelEncoder. 5.
Separate the features and target (median_house_value). 6. Split the
data into training and testing sets (80/20). 7. Scale features using
StandardScaler. 8. Train Linear Regression, Ridge Regression, and
Lasso Regression models. 9. Use GridSearchCV with five-fold
cross-validation to tune Ridge and Lasso alpha values. 10. Evaluate
predictions using MSE, RMSE, and R². 11. Visualize predicted vs. actual
values and residual distributions.

Results recorded in the notebook:

  Model                              MSE        RMSE       R²
  Linear Regression     5,055,025,116.17   71,098.70   0.6142
Ridge Regression      5,050,126,954.37   71,064.25   0.6146
Lasso Regression      5,054,098,843.02   71,092.19   0.6143

Ridge Regression achieved the highest R² among the three results
recorded (approximately 0.6146).
3. NLP Sentiment Classification
Notebook: NLP_classification_pipeline_notebook.ipynb

Objective: Classify customer/product reviews into positive,
negative, or neutral sentiment.

Dataset: sentiment_reviews (1).csv
The notebook preview contains 10 review records with review_id,
review_text, and sentiment columns.

Workflow: 1. Load review data using Pandas. 2. Clean review text by
removing punctuation, converting text to lowercase, tokenizing, removing
English stop words, and applying WordNet lemmatization. 3. Split the
cleaned text and sentiment labels into training and testing sets (50/50,
random_state=42). 4. Convert text into numerical features using
TfidfVectorizer (max_features=5000). 5. Train Multinomial Naive
Bayes and Logistic Regression models. 6. Check model scores and use the
trained Naive Bayes model for the sample review prediction shown in the
notebook.

Result recorded in the notebook: - Multinomial Naive Bayes training
score: 1.00 (100%) - Multinomial Naive Bayes test accuracy: 0.60
(60%)

The dataset preview contains only 10 examples, so this result is based
on a very small sample and is not sufficient to establish real-world
model performance.



Technologies & Libraries
- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- NLTK
