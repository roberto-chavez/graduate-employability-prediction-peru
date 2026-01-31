# Graduate Employability Prediction using Machine Learning

## Project Overview
This project aims to identify the key factors determining the employability of university graduates. It features a comparative analysis between **Peruvian national data** (INEI - National Survey of Graduates) and **International datasets** (Kaggle). 

Using a mix of **Artificial Neural Networks (MLP)** and ensemble methods like **XGBoost** and **Random Forest**, we classified graduates into three categories: Unemployed, Non-Related Employment, and Career-Related Employment.

## Key Features
* **Data Source:** INEI (Peru) +10,000 records & Kaggle Graduate Datasets.
* **Techniques:** SMOTE for class balancing, StandardScaler, One-Hot Encoding, and Word Embeddings.
* **Models:** Multi-Layer Perceptron (MLP), Random Forest, XGBoost.

## Results & Insights

### 1. The Peruvian Context (INEI Data)
* **The "Experience" Factor:** In Peru, having a professional degree or being on the "Honor Roll" (Cuadro de Mérito) showed a lower correlation with employment than having **Internship Experience**.
* **Model Performance:** XGBoost combined with SMOTE outperformed Neural Networks for national data due to the high noise and class imbalance in the survey.

### 2. International Comparison
* **Model Accuracy:** Reached **98.95% Accuracy** on international datasets where "Soft Skills" and "Communication Scores" were available.
* **Critical Gap:** The study highlights that the lack of "Soft Skill" metrics in the Peruvian national survey limits the predictive power of local models, suggesting universities should track these variables.

| Model | Scenario | Accuracy | Key Metric |
| :--- | :--- | :--- | :--- |
| **XGBoost + SMOTE** | Peru (National) | ~60% (Recall focused) | Best for Imbalance |
| **MLP (Neural Net)** | International | 98.95% | Best for Structured Data |

## Key Findings

* **Practical Experience > Academic Metrics:** In the Peruvian market, internship experience is a more reliable predictor of employability than a high CGPA or belonging to the "Honor Roll" (Cuadro de Mérito).
* **The "Soft Skills" Gap:** International models achieved near-perfect accuracy (98.95%) by including communication and soft skill ratings—variables currently missing in the Peruvian national survey (INEI).
* **Model Suitability:** Ensemble methods (XGBoost/Random Forest) proved more resilient than Neural Networks when dealing with "noisy" and imbalanced socio-economic data from national surveys.
* **The "Non-Related" Employment Trap:** A significant portion of graduates find employment quickly, but not in fields related to their professional training, highlighting a mismatch between academic curricula and market demand.

## Ethical Considerations
We addressed the risk of **algorithmic bias**. In employment models, we prioritized **Recall** over Precision for the "Unemployed" class to ensure that vulnerable students are not overlooked by support programs.

## Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib, Seaborn, TensorFlow/Keras.

## Repository Structure

* `data/`: Documentation regarding the INEI and Kaggle dataset schemas.
* `notebooks/`: Contains the end-to-end data science pipeline (Cleaning, EDA, and Modeling).
* `reports/`: Includes the final technical paper `final_report_graduate_employability_peru.pdf` and visual assets.

## Authors
* Roberto Chavez
* Alvaro Escudero
* Cristian Millan
* Leoncio Villanueva
* Miguel Villegas
