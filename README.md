🛒 Amazon Customer Review Sentiment Analysis

This project applies Natural Language Processing (NLP) techniques to analyze and classify customer sentiments from Amazon product reviews. Using pre-trained transformer models, the goal is to automatically determine whether a review is Positive, Negative, or Neutral, enabling businesses and researchers to better understand customer satisfaction and feedback patterns.
🎯 Objective

    Clean, preprocess, and analyze large-scale Amazon customer review datasets.

    Use pre-trained sentiment analysis models to classify review sentiments.

    Visualize sentiment trends and gain actionable business insights.

    Serve as a foundation for scalable sentiment pipelines for e-commerce data.

📦 Dataset

    Source: Amazon Customer Review Dataset (via Kaggle or public sources)

    Size: ~20,000+ reviews

    Fields Used: reviewText, overall (rating), summary, reviewTime

🧹 Preprocessing Steps

    Text normalization: lowercasing, punctuation removal, stopword filtering

    Removal of null entries and duplicates

    Tokenization for model input

    Optional lemmatization and spell correction for cleaner NLP input

🤖 Sentiment Classification

Sentiment analysis was performed using state-of-the-art transformer models:
Models Used

    cardiffnlp/twitter-roberta-base-sentiment

    nlptown/bert-base-multilingual-uncased-sentiment (optional alternative)

    HuggingFace pipeline for zero-shot or classification-based tagging

Each review is labeled as:

    Positive

    Neutral

    Negative

Model Workflow

    Load and tokenize the review text

    Run through pre-trained transformer pipeline

    Store results in a new column (predicted_sentiment)

📊 Visualizations

Key visual insights were generated using matplotlib and seaborn:

    Sentiment distribution bar chart

    Sentiment trends across review rating scores

    Word clouds for each sentiment category

    Sample review predictions and sentiment confidence scores

(Note: Plots and result files are in /outputs or notebooks folder.)
🧰 Tech Stack
Component	Tools/Frameworks
Programming	Python
NLP Models	HuggingFace Transformers
Libraries	pandas, matplotlib, seaborn, transformers, nltk, scikit-learn
Jupyter Notebooks	For interactive exploration and reporting
Environment	Google Colab / JupyterLab compatible

📝 Sample Output

Review: "This product is amazing! Worked better than expected."
→ Predicted Sentiment: Positive (Score: 0.91)

Review: "Item arrived late and packaging was poor."
→ Predicted Sentiment: Negative (Score: 0.87)

🚀 How to Run

    Clone the repository

git clone https://github.com/hrhaditya/amazon-customer-review-sentiment-analysis
cd amazon-customer-review-sentiment-analysis

Install dependencies

pip install -r requirements.txt

Run notebook or Python scripts

    jupyter notebook notebooks/sentiment_analysis_roberta.ipynb

📌 Future Improvements

    Fine-tuning on domain-specific review data (electronics, books, etc.)

    Deploying as a web API using FastAPI

    Integrating confidence thresholds for model predictions

    Expanding to multilingual sentiment detection

