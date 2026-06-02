# Resume-Category-Classification-
Resume Category Classification using Machine Learning An end-to-end NLP project that classifies resumes into 25 job categories using TF-IDF vectorization and Linear SVM. The pipeline includes text preprocessing, feature extraction, model training, and deployment-ready pickle serialization for real-time predictions.

# Resume Category Classification-
A machine learning pipeline to automatically classify resumes into job categories using NLP and SVM.
Overview
This project builds a resume classification system that reads raw resume text and predicts the job category (e.g., Data Science, Web Designing, HR, Mechanical Engineer). It demonstrates a complete ML workflow from data cleaning to model deployment.
Dataset
File: UpdatedResumeDataSet.csv
Size: 962 rows × 2 columns
Columns: Category (target label), Resume (raw text)
Classes: 25 job categories
Preprocessing: Deduplication applied before train-test split to prevent data leakage

# Projec stucture-
├── UpdatedResumeDataSet.csv    # Raw dataset
├── resume_classifier.ipynb     # Main notebook (your code)
├── Tfidf.pkl                   # Saved TF-IDF vectorizer
├── svm.pkl                     # Saved LinearSVC model
├── label_encoder.pkl           # Saved Label Encoder
└── README.md                   # This file

# Model Performance-
Algorithm: Linear Support Vector Classification (LinearSVC)
Class Handling: class_weight='balanced' for imbalanced categories
Features: 5000 TF-IDF unigrams + bigrams
Evaluation: Stratified split ensures fair representation of all 25 classes

# Key Design Decisions-
Deduplication before split: Critical to avoid data leakage from duplicate resumes appearing in both train and test sets
Balanced class weights: Compensates for category imbalance (e.g., Java Developer 84 vs Advocate 20)
N-gram range (1,2): Captures both single terms and phrase patterns common in resumes
Sublinear TF: Reduces impact of very frequent terms
License
# Open source for educational and portfolio purposes.
