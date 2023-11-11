from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


from googletrans import Translator
import random

# From https://spotintelligence.com/2022/12/19/text-similarity-python/
def text_similarity(text1, text2):
    # Tokenize and lemmatize the texts
    tokens1 = word_tokenize(text1)
    tokens2 = word_tokenize(text2)
    lemmatizer = WordNetLemmatizer()
    tokens1 = [lemmatizer.lemmatize(token) for token in tokens1]
    tokens2 = [lemmatizer.lemmatize(token) for token in tokens2]

    # Remove stopwords
    stop_words = stopwords.words('english')
    tokens1 = [token for token in tokens1 if token not in stop_words]
    tokens2 = [token for token in tokens2 if token not in stop_words]

    # Create the TF-IDF vectors
    vectorizer = TfidfVectorizer()
    vector1 = vectorizer.fit_transform(tokens1)
    vector2 = vectorizer.transform(tokens2)

    # Calculate the cosine similarity
    similarity = cosine_similarity(vector1, vector2)

    return similarity

if __name__ == "__main__":
    # Open sample data
    file = open('Anthakaran.txt','r',encoding='utf-8')
    content = file.readlines()

    while(1):
        # Get random word from random line in text
        index = random.randint(0,len(content)-1)
        line = [i for i in content[index].strip('\n').split(" ") if i != '' and i != ' ']
        word = line[random.randint(0,len(line)-1)]
        
        # Translate word using google translate
        translator= Translator()
        translation = translator.translate(word).text
        
        # Print results 
        console_input = input(f"Translate to English: \"{word}\"\n")
        
        print(f"\nYour translation:\n{console_input}")
        
        print(f"\nActual translation:\n{translation}\n")
    
# Anthakaran.txt from Aesthetics Text Corpus Dataset
# https://github.com/gayatrivenugopal/Hindi-Aesthetics-Corpus