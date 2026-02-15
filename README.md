# Machine Learning Model Evaluation Project

## 1. Problem Statement
The objective of this project is to build and evaluate multiple machine learning models on a given dataset and compare their performance using standard evaluation metrics. The models are trained and tested to identify the best-performing algorithm based on classification performance.

---

## 2. Dataset Description
The Global Cars Enhanced dataset provides information on various car models from different manufacturers around the globe. The dataset includes various attributes related to the car, such as numerical and categorical attributes.
Key Features:
•	Manufacturer & Model – Brand and model name of the car
•	Year – Manufacturing year
•	Fuel Type – Petrol, Diesel, Electric, Hybrid, etc.
•	Transmission – Manual or Automatic
•	Engine Details – Engine size, horsepower, torque, etc.
•	Mileage/Efficiency – Fuel economy or range
•	Vehicle Type/Body Style – Sedan, SUV, Hatchback, etc.
•	Price – Market price of the vehicle (usually the target or analysis variable)
Purpose:
This dataset is appropriate for machine learning and data analysis activities such as:
•	Price prediction (Regression)
•	Vehicle category classification
•	Market trend analysis
•	Feature importance and comparison studies


---

## 3. Models Used
The following machine learning models are implemented and evaluated:

1. Logistic Regression  
2. Decision Tree  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

Each model is evaluated using the following metrics:
- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)

---

## 4. Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------------|---------|-----|-----------|--------|----------|-----|
| Decision Tree | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBoost (Ensemble) | 0.9667 | 0.9956 | 0.9706 | 0.9667 | 0.9665 | 0.9569 |
| Random Forest (Ensemble) | 0.9167 | 0.9755 | 0.9212 | 0.9167 | 0.9163 | 0.8903 |
| Logistic Regression | 0.9000 | 0.9814 | 0.9008 | 0.9000 | 0.8999 | 0.8668 |
| KNN | 0.6000 | 0.7897 | 0.5800 | 0.6000 | 0.5762 | 0.4739 |
| Naive Bayes | 0.5667 | 0.8863 | 0.7009 | 0.5667 | 0.5416 | 0.4517 |

## 5. Observations on Model Performance

| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| Decision Tree | Achieved perfect scores on all metrics. This may indicate validation on a small dataset or potential overfitting. |
| XGBoost (Ensemble) | Second-best performance with very high AUC and MCC, showing strong generalization and stability. |
| Random Forest (Ensemble) | Good performance with balanced precision and recall, indicating reliable predictions. |
| Logistic Regression | Performs well with high AUC, suitable for linearly separable data but slightly lower than ensemble methods. |
| KNN | Moderate performance; sensitive to feature scaling and data distribution. |
| Naive Bayes | Lowest overall performance due to strong independence assumptions among features. |

---

## 6. Conclusion
Although the Decision Tree shows perfect performance, ensemble methods like XGBoost and Random Forest are generally more reliable for real-world deployment because they reduce overfitting and provide better generalization.

---

## 7. How to Run the Project
1. Place the dataset in the `data/` folder.
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt