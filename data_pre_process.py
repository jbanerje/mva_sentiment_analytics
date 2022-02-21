# Core packages
import string
import re
# Spacy Packages
import spacy
nlp = spacy.load('en_core_web_md')
lemmatizer = nlp.get_pipe("lemmatizer")
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
        
        word_list_lemmatized = [word.lemma_ for word in nlp(text) if not (word.like_num or word.is_stop or word.is_punct or word.is_space or len(word)==1)]
        
        # word_list = text.split()
        
        # # Word List after removing stop words
        # word_list_clean = [word for word in word_list if word not in stopwords]
        
        # # Lemmatize the list
        # sentence_after_cleaning = ' '.join(word_list_clean)
        # doc = nlp(sentence_after_cleaning)
        # word_list_lemmatized = [token.lemma_ for token in doc]
        
        return text, list(set(word_list_lemmatized))