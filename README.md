# Predicting Heart Disease - Kaggle Playground Series S6E2 Competition

**Group 9 - Probability and Statistics (F)**
|    NRP     |      Name      |
| :--------: | :------------: |
| 5025241086 |  Callista Fidelya Roba Gultom |
| 5025241180 | Dilbina Windi Azahra |
| 5025241199 | Isabella Sienna Sulisthio |
| 5025241243 | Najma Lail Arazy |

Dataset: [Predicting Heart Disease - Kaggle](https://www.kaggle.com/competitions/playground-series-s6e2/data)

**Bayes’ Theorem**

Bayes’ Theorem is a mathematical formula used to determine the probability of an event based on prior knowledge of related conditions. It describes how the probability of a hypothesis changes when new evidence is observed.​

``P(A∣B)= P(B∣A)P(A) / P(B)``​

P(A | B) → Posterior probability: the probability of event A occurring given that B is true.

P(B | A) → Likelihood: the probability of observing evidence B if A is true.

P(A) → Prior probability: the initial probability of event A before seeing evidence.

P(B) → Evidence probability: the probability of observing B.

Bayes’ Theorem is widely used in statistics, machine learning, medical diagnosis, and decision-making systems because it allows probabilities to be updated as new data becomes available.

**Application of Bayes’ Theorem in the Naive Bayes Algorithm**

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
yg blm:
- penjelasan singkat mengenai Teori Bayes
- penjelasan bagaimana Teori Bayes digunakan dalam Algoritma Naive Bayes
- penjelasan dataset yang digunakan
- hasil eksekusi Naive Bayes, dilengkapi evaluasi akurasi

Dianjurkan repository tersebut di-fork ke GitHub anggota grup. Dianjurkan menggunakan bahasa inggris dan menyajikan penjelasan secara visual
