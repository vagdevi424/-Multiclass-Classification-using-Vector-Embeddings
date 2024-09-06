# -Multiclass-Classification-using-Vector-Embeddings

Problem Statement:
The objective of this project is to develop a predictive model for multiclass classification using vector embeddings. The dataset contains vector embeddings of 54,000 rows, which are categorized into 10 different classes. The model aims to classify each set of vector embeddings into the correct class, which is a common task in natural language processing (NLP) applications, especially in Large Language Models (LLMs), where vector embeddings capture the semantic relationships between words.

Dataset Description:
The dataset consists of vector embeddings, which are numerical representations capturing relationships between words, phrases, or sentences. These embeddings are arranged in a matrix of 384 features, with an additional column representing the target class (from 1 to 10). The goal is to use these features to classify sentences into one of the 10 categories.

Approach and Key Steps:
Data Preprocessing:

The data is loaded and preprocessed by converting the array of embeddings into features, resulting in 384 features for each record.
Basic data analysis is conducted to understand the distribution and shape of the dataset, confirming that no missing values are present.
Visualization:

Visualizations such as histograms and scatter plots are used to understand feature distributions and relationships between different features. These plots help in identifying patterns, outliers, and possible correlations between the features.
Principal Component Analysis (PCA):

PCA with 100 components is performed to reduce the dimensionality of the data while retaining most of the variance.
The top 2 components are visualized, and clustering analysis is performed using K-Means, showing distinct clusters in the reduced feature space.
Feature Engineering:

Features are scaled using StandardScaler for more efficient processing.
PCA-transformed features are used as input for classification models.
Modeling:

Multiple classifiers are trained and evaluated, including:
Logistic Regression
K-Nearest Neighbors (KNN)
XGBoost
Hyperparameter tuning is performed using GridSearchCV to optimize each model.
Cross-validation is applied to assess model performance based on metrics such as accuracy, precision, recall, and F1 score.
Model Evaluation:

The Logistic Regression model performed best on the validation set, achieving high accuracy (72.3%), precision (71.8%), recall (72.3%), and F1 score (71.8%).
The best model is selected and evaluated on a separate test set, with predictions saved in a CSV file.
Final Evaluation:

The final model is used to predict labels on an unseen test set and the results are submitted in a CSV file for external evaluation.
Conclusion:
The project successfully implements a multiclass classification model using vector embeddings. By leveraging PCA for dimensionality reduction, feature scaling, and model optimization, the model achieves robust performance on both the validation and test sets. The Logistic Regression model was found to be the most effective for this dataset.
