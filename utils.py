import spacy
nlp = spacy.load("en_core_web_sm")
from spacy_streamlit import visualize_ner, visualize_tokens, visualize_textcat
# https://github.com/explosion/spacy-streamlit

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

def read_neg_word_dict(pre_processed_text):
    
    ''' Negative Polarity '''        
    
    file_neg_lex = open("./static/negative-words.txt", "r")
    neg_lex_list = file_neg_lex.read().split()
        
    neg_feedback_words = [neg_word for neg_word in pre_processed_text if neg_word in neg_lex_list]
        
    if len(neg_feedback_words) > 0 :
        return list(set(neg_feedback_words))
    else:
        return []

def read_pos_word_dict(pre_processed_text):
    
    ''' Positive Polarity '''        
    
    file_neg_lex = open("./static/positive-words.txt", "r")
    pos_lex_list = file_neg_lex.read().split()
        
    pos_feedback_words = [pos_word for pos_word in pre_processed_text if pos_word in pos_lex_list]
        
    if len(pos_feedback_words) > 0 :
        return list(set(pos_feedback_words))
    else:
        return []

def pos_tagging(text):
    
    ''' Function Returs Proper Noun, Numbers and Nounns '''
    
    entity_identification = []
    
    cars_file = open("./static/porsche_cars.txt", "r")
    cars_file_list = cars_file.read().lower().split()
    
    doc = nlp(text.lower())
    #visualize_tokens(doc, attrs=["text", "pos_", "dep_", "ent_type_"])
    
    pos_numbers  = list(set([token.text for token in doc if token.pos_ in ['NUM']]))
    pos_keywords = list(set([token.text for token in doc if token.pos_ in ['PROPN', 'NOUN']]))
    
    # Extracting the Year
    for numbers in pos_numbers:
        try:
            if ( int(numbers) >=  1990) and ( int(numbers) <= 2050 ):
                entity_identification.append(numbers)
            if (numbers in cars_file_list) and (int(numbers) > 900):
                entity_identification.append(numbers)
                
        except Exception as e:
            pass
               
    # Extracting the Model
    for items in pos_keywords:
        if items in cars_file_list:
            entity_identification.append(items)
                    

    
    return entity_identification


def get_noun_chunks(text):
    
    ''' Funcxtion Returns Noun Chunks '''
    
    doc = nlp(text)
    
    return [chunk.text for chunk in doc.noun_chunks]

def display_name_entity_viz(text):
    doc = nlp(text)
    # print(nlp.get_pipe("ner").labels)
    # visualize_ner(doc, labels=nlp.get_pipe("ner").labels)
    visualize_ner(
                doc, 
                labels=['PERSON', 'DATE', 'GPE', 'ORG', 'EVENT', 'FAC', 'LOC', 'PRODUCT'], 
                show_table=False,
                title="Keyword Vizualization"
            )
    
    return