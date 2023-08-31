from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Read the text file containing movie data
with open(r'C:\Users\HP\Desktop\coding\python\internship\Genre Classification Dataset\train_data.txt', 'r', encoding='utf-8') as file:
    movie_data = file.readlines()

# Separate serial_numbers, movie titles, genres, and descriptions
serial_numbers = []
movies = []
genres = []
descriptions = []
for line in movie_data:
    movie_parts = line.strip().split(":::")
    serial_numbers = [0]
    title = movie_parts[1]
    genre = movie_parts[2].strip().lower()
    description = movie_parts[3]
    serial_numbers.append(serial_numbers)
    movies.append(title)
    genres.append(genre)
    descriptions.append(description)

# Convert genre labels to numerical values
genre_mapping = {'action': 0, 'comedy': 1, 'drama': 2, 'sci-fi': 3, 'horror': 4,
                  "thriller": 5, 'adult': 6, 'documentary': 7, 'reality-tv': 8,
                    'sport': 9, 'crime': 10, 'animation': 11, 'fantasy': 12, 'short': 13,
                      'music': 14, 'adventure': 15, 'talk-show': 16, 'western': 17, 'family': 18,
                        'mystery': 19, 'history': 20,'news': 21, 'biography': 22, 'romance': 23,
                          'game-show': 24, 'musical': 25, 'war': 26}
numerical_genres = [genre_mapping[genre] for genre in genres]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(descriptions, numerical_genres, test_size=0.2, random_state=42)

# Convert movie descriptions to numerical features using TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Create a Naive Bayes classifier
classifier = MultinomialNB()

# Train the classifier
classifier.fit(X_train_vec, y_train)

# Predict genres for the test data
predictions = classifier.predict(X_test_vec)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")

# Generate a classification report
report = classification_report(y_test, predictions, target_names=list(genre_mapping.keys()))
print("Classification Report:\n", report)
