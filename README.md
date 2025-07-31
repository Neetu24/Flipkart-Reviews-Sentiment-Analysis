# 💬 Flipkart Reviews Sentiment Analysis using Python

## 🎯 Objective

Build a machine learning model to analyze Flipkart product reviews and classify them as **Positive** or **Negative** based on sentiment. This helps businesses gain insights into customer satisfaction and product quality.

---

## 📂 Dataset

- **Name:** Flipkart Product Reviews Dataset  
- **Content:** Text reviews with associated sentiment labels  
- **Format:** CSV/Text-based (Assumed structure: `review`, `sentiment`)  

---

## 🛠️ Tools & Libraries Used

- **Python**
- **Pandas** – Data manipulation
- **scikit-learn** – ML algorithms, preprocessing, evaluation
- **Matplotlib/Seaborn** – Data visualization
- **WordCloud** – Visualization of common words
- **Regex/NLTK (optional)** – Text cleaning and stopword removal
- **Warnings** – To suppress unnecessary logs

---

## 🔁 Project Workflow

### 1. Data Loading
- Load the Flipkart reviews dataset using `pandas`.
- Inspect data structure, null values, duplicates.

### 2. Data Preprocessing
- Drop null or duplicate entries.
- Convert text to lowercase.
- Remove:
  - Stopwords
  - Punctuation
  - Special characters
- Encode sentiment labels:
  - `Positive` → 1
  - `Negative` → 0
- Convert text into numerical features using **TF-IDF vectorization**.
- Split into train-test sets (80%-20%).

### 3. Exploratory Data Analysis (EDA)
- Plot **sentiment distribution** using `countplot`.
- Create **WordClouds** for both positive and negative reviews.
- Explore correlation between **review length** and **sentiment polarity**.

### 4. Model Training
- Train and compare the following ML models:
  - Logistic Regression
  - Naive Bayes
  - Random Forest Classifier
  - Support Vector Machine (SVM)

### 5. Model Evaluation
- Evaluate all models on test data using:
  - **Accuracy**
  - **Precision, Recall, F1-score**
  - **Confusion Matrix**
- Choose the best-performing model.

### 6. Prediction
- Test the final model on **new/unseen reviews** to predict their sentiment.

---

## 🧪 Sample Results

Example:
Review: "The product quality is amazing and delivery was fast!"
Predicted Sentiment: Positive ✅

Review: "Worst product ever, totally waste of money!"
Predicted Sentiment: Negative ❌

---

📁 File Structure
📦 flipkart-sentiment-analysis/
 ┣ 📜 sentiment_analysis.py          ← Main ML script
 ┣ 📒 sentiment_analysis.ipynb       ← Jupyter notebook (optional)
 ┣ 📂 data/
 ┃ ┗ 📄 flipkart_reviews.csv         ← Dataset
 ┣ 📊 outputs/
 ┃ ┣ 📈 sentiment_distribution.png
 ┃ ┗ ☁️ wordcloud.png
 ┣ 📄 README.md
 ┣ 📄 requirements.txt

