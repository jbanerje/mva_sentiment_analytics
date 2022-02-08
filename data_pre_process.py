# Core packages
import string
import re

# NLP Packages
from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords
stopwords_english = stopwords.words('english')

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

class DataPreProcess:
    
    def __init__(self, text):
        self.text = text
        
    def perform_data_pre_processing(self):
        '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
        
        word_list_clean = []
        
        text = self.text.lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\w*\d\w*', '', text)
        text = re.sub('[‘’“”…]', '', text)
        text = re.sub('\n', '', text)
        
        text = lemmatizer.lemmatize(text) # Lemmatize (Root Words)
        
        text_list = text.split()
        
        for word in text_list:
            if (word not in stopwords_english and  # remove stopwords
                    word not in string.punctuation):  # remove punctuation
                    word_list_clean.append(word)
                
        return word_list_clean