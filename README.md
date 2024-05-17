# SMS Spam Detection Using Support Vector Machines (SVM)

This repository contains the code and report for an SMS Spam Detection system using Support Vector Machines (SVM). The project was completed as part of the ITCS 6150/8150 Intelligent Systems course. The primary objective of this project is to accurately classify SMS messages as either "spam" or "ham" using machine learning techniques.

## Project Overview

Spam emails and messages pose significant challenges, affecting both communication efficiency and security. This project aims to develop a reliable spam detection model using SVM, a robust method for binary classification tasks. The dataset used for this project is the SMS Spam Collection dataset from the UCI Machine Learning Repository.

## Repository Contents

* Report.pdf: A detailed report of the project, including objectives, methodology, results, and conclusions.
* Spam_Detection.ipynb: Jupyter Notebook containing the complete code for data preprocessing, model training, evaluation, and prediction.

## Getting Started
### Prerequisites

To run the code in this repository, you will need the following:

* Python 3.x
* Jupyter Notebook
* Necessary Python libraries: pandas, numpy, scikit-learn, nltk, pickle

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/sms-spam-detection.git
   cd sms-spam-detection
   ```

2. Install the required libraries:
   
   ```bash
   pip install pandas numpy scikit-learn nltk pickle
   ```

3. Download the dataset:

The dataset can be directly downloaded from the UCI Machine Learning Repository.

### Running the Project

1. Open the Jupyter Notebook:

   ```bash
    jupyter notebook Spam_Detection.ipynb
   ```
   
2. Run the cells in the notebook to execute the code:

The notebook includes steps for:
* Data preprocessing
* Text vectorization using CountVectorizer and TfidfTransformer
* Model training using SVM
* Model evaluation
* Custom input testing

3. Test with Custom Inputs:

You can test the trained model with custom SMS messages to check if they are classified as spam or ham.

## Detailed Steps
### Step 1: Data Preprocessing

* Load the SMS Spam Collection dataset.
* Convert text labels to numeric labels using LabelEncoder.
* Perform text preprocessing, including tokenization, removal of stopwords, and stemming.

### Step 2: Split the Dataset

* Split the data into training and test sets.

### Step 3: Text Vectorization

* Use CountVectorizer to convert text data into numerical features.
* Apply TfidfTransformer to normalize the features.

### Step 4: Model Training

* Train an SVM model with a linear kernel using the preprocessed data.

### Step 5: Model Evaluation

* Evaluate the model using the test dataset and generate a classification report and confusion matrix.

### Step 6: Save the Model

* Save the trained model, vectorizer, and TF-IDF transformer for future predictions.

### Step 7: Custom Input Testing

* Load the saved model, vectorizer, and transformer.
* Preprocess custom SMS messages and use the trained model to predict if they are spam or ham.

## Conclusion

The SVM-based SMS spam detection model achieved an impressive accuracy of 99.1%. The model showed high precision and recall for both classes, making it an effective tool for spam detection. Future work can focus on improving the model's adaptability to new spam techniques and minimizing false negatives.

## Sources

* [SMS Spam Collection Dataset - UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
* Trivedi, Shrawan Kumar. "A study of machine learning classifiers for spam detection." BML Munjal University, Gurgaon, Haryana, India.[Link](https://ieeexplore.ieee.org/abstract/document/7743279/authors#authors)
* Agboola, Olubodunde et al. "Spam Detection Using Machine Learning and Deep Learning." Ph.D. Dissertation.[Link](https://dl.acm.org/doi/book/10.5555/aai30276699)

## Acknowledgement

Special thanks to the UCI Machine Learning Repository for providing the dataset and to the authors of the referenced studies for their valuable insights into spam detection.

For any questions or suggestions, feel free to open an issue or contact the authors:

* Nishan Dhakal
* Afif Mujibur-Rahman
