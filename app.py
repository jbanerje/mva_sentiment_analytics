# Core Pkgs
import streamlit as st

# EDA Pkgs
import pandas as pd 
import numpy as np 
from datetime import datetime

# Vizualization Package 
import matplotlib.pyplot as plt

# In-built Class
from get_sentiments import *
from get_emotions   import *
from data_pre_process import *
from utils import *


def extract_key_values_from_dict(input_dict):
    
    '''  Function to extract keys & values from dictionary for plotting '''
    labels = []
    sizes  = []

    for key, value in input_dict.items():
        if value > 0:
            labels.append(key)
            sizes.append(value)
        
    return labels, sizes
    
    
def sentiment_pie_chart(sizes, labels):
    
    ''' Function For Pie Chart '''
    
    fig1, ax1   = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    
    return fig1

           
def load_sentiment_analysis_ui():
    
    # Page Setup
    st.set_page_config(
    page_title="CUSTOMER SENTIMENT ANALYSIS",
    page_icon='./images/sentiment_analysis_fav.png',
    layout="centered",
    initial_sidebar_state="auto")
    
    # Real Time Search Box
    with st.form(key='emotion_clf_form'):
        
        st.header('Analyze Your Sentence')
        
        # Grab Raw Text
        raw_text            = st.text_area("")
        submit_text         = st.form_submit_button(label='Submit')
        
    if submit_text:
        
        # Show Spacy Vizualizer
        display_name_entity_viz(raw_text)
                            
        # Build 2 sections in the UI
        col1,col2  = st.columns(2)
        
        # Pre-Process Text - Remove unwanted chacaracters, stem, lemmatize etc
        pre_processed_text      = DataPreProcess(raw_text)
        clean_text_list         = pre_processed_text.perform_data_pre_processing()
        clean_text_str          = ' '.join(clean_text_list)
        
        # Get Sentiments
        sentiment_polarity      = analyzeSentiments(raw_text)       
        
        # Get Emotions
        emotions_info           = Emotions(raw_text)
        
        with col1:
            
            # Polarity Extractions
            polarity_info       = sentiment_polarity.get_sentiment_polarity()
            polarity_df         = sentiment_polarity.get_sentiment_data()
            
            st.info(f'Sentiment - {polarity_info}')
            
            if polarity_info  == 'Negative':
                polarity_word_list = read_neg_word_dict(clean_text_list)
            else:
                polarity_word_list = read_pos_word_dict(clean_text_list)
                    
            st.pyplot(sentiment_pie_chart(list(polarity_df['Score']), list(polarity_df['Sentiment'].unique())))
            
            # Code Block For Entity Information
            entity_information  = pos_tagging(raw_text)
            entity_information = [entity.capitalize() for entity in entity_information]
            
            st.info('Entity Information')
            
            if len(entity_information) > 0 :
                entity_information_str = ' '.join(entity_information)
                st.markdown(f""" ###### {entity_information_str}""")
            else:
                st.markdown(f""" Not Available """)
            
            # Code Block For Customer Feedback
            st.info('Customer Feedback')
            
            if len(polarity_word_list) > 0 :
                for pol_word in polarity_word_list:
                    st.markdown(f""" * ###### {pol_word.capitalize()}""")
            else:
                st.write(f""" Not Available """)
                
                
        with col2 :
            
            # Code Block For Entity information
            labels, sizes = extract_key_values_from_dict(emotions_info.extract_emotions())
            st.info('Emotions')
            st.pyplot(sentiment_pie_chart(sizes, labels))
            
            # Code Block For Focus Area
            focus_area_from_static      = identify_focus_areas(clean_text_list)
            
            st.info('Focus Area')
            
            if len(focus_area_from_static) > 0 :
                for focus_word in focus_area_from_static:
                    st.markdown(f""" * ###### {focus_word.capitalize()}""")
            else:
                st.markdown(f""" Not Available """)
            
            # # Code Block for Aditional references
            # focus_area_from_noun_chunks = get_noun_chunks(clean_text_str)
            # st.info('Additional Tags')
            
            # if len(focus_area_from_noun_chunks) > 0 :
            #     for addtnl_tag in focus_area_from_noun_chunks:
            #         st.markdown(f""" - {addtnl_tag.capitalize()}""")
            # else:
            #     st.markdown(f""" Not Available """)
        
if __name__ == "__main__":
    load_sentiment_analysis_ui()