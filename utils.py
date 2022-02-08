import spacy
nlp = spacy.load("en_core_web_sm")

def identify_focus_areas(pre_processed_text):
    
    ''' Function to identify problem areas '''
    problem_area_list = []
        
    file_car_parts = open("./static/car_parts.txt", "r")
    car_parts_list = file_car_parts.read().split()
    
    for word in pre_processed_text:
        if word in car_parts_list:
            problem_area_list.append(word)
        
    if len(problem_area_list) > 0 :
        return list(set(problem_area_list))
    else:
        return ''

def read_neg_pos_word_dict(polarity_info, pre_processed_text):
    
    ''' This file read the negative & postive word from files '''
    
    neg_feedback_words = []
    pos_feedback_words = []
        
    if polarity_info == 'Negative':
        
        file_neg_lex = open("./static/negative-words.txt", "r")
        neg_lex_list = file_neg_lex.read().split()
        
        for neg_word in pre_processed_text:
            if neg_word in neg_lex_list:
                neg_feedback_words.append(neg_word)
        
        if len(neg_feedback_words) > 0 :
            return list(set(neg_feedback_words))
        else:
            return 'Could not find any matching words'
    
    else:
        
        file_pos_lex = open("./static/positive-words.txt", "r")
        pos_lex_list = file_pos_lex.read().split()

        for pos_word in pre_processed_text :
            if pos_word in pos_lex_list:
                pos_feedback_words.append(pos_word)
        
        if len(pos_feedback_words) > 0 :
            return list(set(pos_feedback_words))
        else:
            return 'Could not find any matching words'

def pos_tagging(text):
    
    ''' Function Returs Proper Noun, Numbers and Nounns '''
    doc = nlp(text)
    pos_keywords = list(set([token.text for token in doc if (token.pos_ == 'PROPN')]))
    return pos_keywords


def get_noun_chunks(text):
    
    ''' Funcxtion Returns Noun Chunks '''
    
    doc = nlp(text)
    return [chunk.text for chunk in doc.noun_chunks]