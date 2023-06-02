import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from newspaper import Article
import joblib

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset
data = pd.read_csv('modnews.csv')

# Function to extract clean text from the article URL
def extract_text(url):
    print(url)
    article = Article(url)
    article.download()
    article.parse()
    return article.text

# Extract the clean text from the article URLs and add it as a new column
data['extracted_text'] = data['url'].apply(extract_text)

# Preprocessing and feature extraction
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())
    
    # Remove punctuation
    table = str.maketrans('', '', string.punctuation)
    tokens = [token.translate(table) for token in tokens]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back into a string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

data['processed_text'] = data['extracted_text'].apply(preprocess_text)

# Split the data into features and target
X = data[['date', 'subject', 'text', 'title', 'processed_text']]
y = data['sentiment'].apply(lambda x: x['class'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps for different feature types
preprocessor = ColumnTransformer([
    ('text', TfidfVectorizer(), 'text'),
    ('categorical', LabelEncoder(), ['subject']),
])

# Define the model pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict the sentiment on the testing data
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#save model as .pkl file
joblib.dump(model, 'sentiment-analysis.pkl')
