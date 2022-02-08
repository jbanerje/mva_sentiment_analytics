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
            
            if  polarity_info   == 'Positive':           
                st.success(f'Sentiment - {polarity_info}')
                word_list = read_neg_pos_word_dict(polarity_info, clean_text_list)
                
            elif polarity_info  == 'Negative':
                st.error(f'Sentiment - {polarity_info}')
                word_list = read_neg_pos_word_dict(polarity_info, clean_text_list)
                
            else:
                st.info(f'Sentiment - {polarity_info}')
                word_list       = read_neg_pos_word_dict(polarity_info, clean_text_list)
                
            st.pyplot(sentiment_pie_chart(list(polarity_df['Score']), list(polarity_df['Sentiment'].unique())))
            
            # Code Block For Entity Information
            entity_information  = pos_tagging(raw_text)
            st.info('Entity Information')
            st.write(entity_information)
            
            # Code Block For Focus Area
            focus_area_from_static      = identify_focus_areas(clean_text_list)
            focus_area_from_noun_chunks = get_noun_chunks(clean_text_str)
            focus_area_from_noun_chunks.append(focus_area_from_static)
            
            st.info('Focus Area')
            st.write(focus_area_from_noun_chunks)
            
        with col2 :
            
            # Code Block For Entity information
            labels, sizes = extract_key_values_from_dict(emotions_info.extract_emotions())
            st.info('Emotions')
            st.pyplot(sentiment_pie_chart(sizes, labels))
            
            # Code Block For Customer Feedback
            st.info('Customer Feedback')
            st.write(word_list)
            
        
if __name__ == "__main__":
    load_sentiment_analysis_ui()