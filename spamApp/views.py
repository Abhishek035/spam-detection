from django.shortcuts import render
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# Create your views here.


def spamApp(request):
    return render(request, 'internship.html')


def result(request):
    message = request.POST.get('message')
    print(message)

    cur_path = os.getcwd()
    print(cur_path)
    csv_path = os.path.join(cur_path, 'spamApp/spam.csv')

    # Load the dataset
    data = pd.read_csv(csv_path)

    # Data preprocessing
    # Encode the 'category' column
    data['Category'] = data['Category'].map({'spam': 1, 'ham': 0})

    # Text data processing
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = tfidf_vectorizer.fit_transform(data['Message'])
    y = data['Category']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42)

    # Model training
    # model = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
    # model = GaussianNB()
    model = MultinomialNB()
    model.fit(X_train, y_train)

    text_vectorized = tfidf_vectorizer.transform([message])
    prediction = model.predict(text_vectorized)
    print(prediction)
    prediction = "Spam !!" if prediction[0] == 1 else "Not Spam !!"

    # Model evaluation
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)

    print("Predicted Email Type:", prediction)
    print("Score is:", accuracy)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return render(request, 'result.html', {'prediction': prediction, 'message': message})
