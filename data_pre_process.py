# Core packages
import string
import re
# Spacy Packages
import spacy
nlp = spacy.load('en_core_web_sm')
stopwords = nlp.Defaults.stop_words

class DataPreProcess:
    
    def __init__(self, text):
        self.text = text
        
    def perform_data_pre_processing(self):
        
        '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
        
        text = self.text.lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\w*\d\w*', '', text)
        text = re.sub('[‘’“”…]', '', text)
        text = re.sub('\n', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation)) # Remove Punctuation
        
        doc = nlp(text)
        
        word_list_clean = [word.lemma_ for word in doc if word not in stopwords]
                
        return word_list_clean