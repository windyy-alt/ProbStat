# Predicting Heart Disease - Kaggle Playground Series S6E2 Competition

**Group 9 - Probability and Statistics (F)**
|    NRP     |      Name      |
| :--------: | :------------: |
| 5025241086 |  Callista Fidelya Roba Gultom |
| 5025241180 | Dilbina Windi Azahra |
| 5025241199 | Isabella Sienna Sulisthio |
| 5025241243 | Najma Lail Arazy |

Dataset: [Predicting Heart Disease - Kaggle](https://www.kaggle.com/competitions/playground-series-s6e2/data)

## Bayes’ Theorem

Bayes’ Theorem is a mathematical formula used to determine the probability of an event based on prior knowledge of related conditions. It describes how the probability of a hypothesis changes when new evidence is observed.​

``P(A∣B)= P(B∣A)P(A) / P(B)``​

P(A | B) → Posterior probability: the probability of event A occurring given that B is true.

P(B | A) → Likelihood: the probability of observing evidence B if A is true.

P(A) → Prior probability: the initial probability of event A before seeing evidence.

P(B) → Evidence probability: the probability of observing B.

Bayes’ Theorem is widely used in statistics, machine learning, medical diagnosis, and decision-making systems because it allows probabilities to be updated as new data becomes available.

## Application of Bayes’ Theorem in the Naive Bayes Algorithm

The Naive Bayes algorithm is a classification method in machine learning that applies Bayes’ Theorem with the assumption that each feature is independent of the others given the class. This assumption simplifies the computation of probabilities when dealing with multiple features. In practice, the algorithm calculates the probability of each class given the observed features and then selects the class with the highest probability as the prediction. Because of its simplicity and efficiency, Naive Bayes is widely used in tasks such as text classification, spam detection, sentiment analysis, and medical data classification.

According to Bayes’ theorem, probability of event A given that event B has already occurred can be calculated using the probabilities of event A and event B and probability of event B given that A has already occurred. Bayes’ theorem is so fundamental and ubiquitous that a field called "bayesian statistics" exists. In bayesian statistics, the probability of an event or hypothesis as evidence comes into play. Therefore, prior probabilities and posterior probabilities differ depending on the evidence.
 
 Naive Bayes methods are a set of supervised learning algorithms based on applying Bayes’ theorem with the “naive” assumption of conditional independence between every pair of features given the value of the class variable. Bayes’ theorem states the following relationship, given class variable y and dependent feature vector $x_1$ through $x_n$ :
 
  $P(y \mid x_1, ..., x_n) = \frac{P(y)\,P(x_1, ..., x_n \mid y)}{P(x_1, ..., x_n)}$ 
        
Using the naive conditional independence assumption that:
$P(x_i \mid y, x_1, ..., x_{i-1}, x_{i+1}, ..., x_n) = P(x_i \mid y)$

for all $i$, this relationship is simplified to: $P(y \mid x_1, ..., x_n) = \frac{P(y)\prod_{i=1}^{n} P(x_i \mid y)}{P(x_1, ..., x_n)}$

Since $P(x_1,...,x_n)$ is constant given the input, we can use the following classification rule: 

$P(y \mid x_1, ..., x_n) \propto P(y)\prod_{i=1}^{n} P(x_i \mid y)$ $=>$ $\hat{y} = \arg\max_y P(y)\prod_{i=1}^{n} P(x_i \mid y)$

The main idea behind the Naive Bayes classifier is to use Bayes' Theorem to classify data based on the probabilities of different classes given the features of the data. It is used mostly in high-dimensional text classification

- The Naive Bayes Classifier is a simple probabilistic classifier and it has very few number of parameters which are used to build the ML models that can predict at a faster speed than other classification algorithms.
- It is a probabilistic classifier because it assumes that one feature in the model is independent of existence of another feature. In other words, each feature contributes to the predictions with no relation between each other.
- Naive Bayes Algorithm is used in spam filtration, Sentimental analysis, classifying articles and many more.

  
## Dataset Description

This project uses the dataset from the **Kaggle Playground Series - Season 6, Episode 2** competition titled *"Predicting Heart Disease"*. The dataset was synthetically generated from a deep learning model trained on the original [Heart Disease Prediction dataset](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data), meaning feature distributions are close to, but not exactly the same as, the original.

---


| File | Description |
|---|---|
| `train.csv` | Training set containing 630,000 rows and 15 columns, including the target label `Heart Disease` |
| `test.csv` | Test set containing 270,000 rows and 14 columns (no target label) |
| `sample_submission.csv` | A sample submission file showing the required output format |

---

### Features

| Feature | Type | Description |
|---|---|---|
| `id` | int | Unique row identifier |
| `Age` | int | Age of the patient in years |
| `Sex` | int | Gender of the patient (1 = male, 0 = female) |
| `Chest pain type` | int | Type of chest pain experienced (1–4) |
| `BP` | int | Resting blood pressure (in mm Hg) |
| `Cholesterol` | int | Serum cholesterol level (in mg/dl) |
| `FBS over 120` | int | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) |
| `EKG results` | int | Resting electrocardiographic results (0, 1, 2) |
| `Max HR` | int | Maximum heart rate achieved |
| `Exercise angina` | int | Exercise-induced angina (1 = yes, 0 = no) |
| `ST depression` | float | ST depression induced by exercise relative to rest |
| `Slope of ST` | int | Slope of the peak exercise ST segment (1–3) |
| `Number of vessels fluro` | int | Number of major vessels colored by fluoroscopy (0–3) |
| `Thallium` | int | Thallium stress test result (3 = normal, 6 = fixed defect, 7 = reversible defect) |
| `Heart Disease` | string | **Target variable** whether heart disease is present (`Presence`) or not (`Absence`) |

---

### Target Variable

The target column is `Heart Disease`, a binary classification label:
- `Absence` (encoded as **0**) = the patient does **not** have heart disease
- `Presence` (encoded as **1**) = the patient **has** heart disease

---


## Model Evaluation Results

### Cross-Validation (Local) vs. Kaggle Public Leaderboard Score

Two Naive Bayes variants were evaluated and compared, both locally via cross-validation and on the Kaggle public leaderboard.

| Model | CV Score (Local) | Kaggle Public Score |
|---|---|---|
| Gaussian Naive Bayes | 0.9230 | 0.9190 |
| Bernoulli Naive Bayes | 0.9452 | 0.9447 |
| **Selected Model** | **BernoulliNB** | **BernoulliNB** |

---

The slight difference between the local cross-validation score and the Kaggle public leaderboard score because:

- **Local CV** evaluates the model on a held-out portion of the **training set**, which was generated from the same synthetic distribution.
- **Kaggle's test set** is a separate, independently sampled portion of data that the model has never seen before, and may have slightly different feature distributions.

The fact that the gap is very small (only 0.0003–0.004) indicates that the model **generalizes well** and is not overfitting.

---

### Why BernoulliNB Outperforms GaussianNB

BernoulliNB consistently outperforms GaussianNB on this dataset for a key structural reason. After applying **OneHotEncoding** to all 8 categorical features, the majority of the feature space becomes **binary (0 or 1) columns**. BernoulliNB is specifically designed to work with binary features, making it a natural and effective fit for this transformed dataset.

GaussianNB, on the other hand, assumes all features follow a **continuous Gaussian (normal) distribution**, an assumption that is violated by the presence of many binary one-hot encoded columns, which explains its slightly lower performance.

---

### Final Model Performance

The final selected model is **BernoulliNB with full preprocessing pipeline**, achieved a Kaggle public leaderboard score of: **0.9447**

For more details, the full implementation including EDA, data preprocessing, feature engineering, ppipeline construction, and model training, please refer to the [Predicting_Heart_Disease_Group_9_Probstat.ipynb](https://github.com/windyy-alt/ProbStat/blob/main/Predicting_Heart_Disease_Group_9_Probstat.ipynb) uploaded in this repository.







