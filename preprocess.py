import os
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

try:
    WordNetLemmatizer()
except LookupError:
    nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path, encoding='latin1')
    print(f"[INFO] Loaded {file_path} with shape: {df.shape}")
    return df

def clean_text(text):
    if pd.isnull(text):
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    text = text.lower()
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

def lemmatize_text(text):
    if pd.isnull(text):
        return ""
    return ' '.join(lemmatizer.lemmatize(word) for word in text.split())

def preprocess_text_for_classification(text):
    return lemmatize_text(clean_text(text))

def preprocess_data(df, task_type="span"):
    df['text'] = df['text'].astype(str)
    if task_type == "classification":
        df['text'] = df['text'].apply(preprocess_text_for_classification)
    elif task_type == "span":
        if 'selected_text' in df.columns:
            df['selected_text'] = df['selected_text'].astype(str)
        else:
            print("[INFO] 'selected_text' column not found — assuming it's a test set without labels.")
    else:
        raise ValueError("task_type must be either 'classification' or 'span'")
    return df

def prepare_data(train_file, test_file, task_type="span", output_dir="data/"):
    train_df = load_data(train_file)
    test_df = load_data(test_file)
    train_df = preprocess_data(train_df, task_type)
    test_df = preprocess_data(test_df, task_type)
    os.makedirs(output_dir, exist_ok=True)
    train_out_path = os.path.join(output_dir, 'preprocessed_train.csv')
    test_out_path = os.path.join(output_dir, 'preprocessed_test.csv')
    train_df.to_csv(train_out_path, index=False)
    test_df.to_csv(test_out_path, index=False)
    print(f"[INFO] Preprocessed data saved:\n  ➤ {train_out_path}\n  ➤ {test_out_path}")
    return train_df, test_df

if __name__ == "__main__":
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'
    prepare_data(train_path, test_path, task_type="span")
