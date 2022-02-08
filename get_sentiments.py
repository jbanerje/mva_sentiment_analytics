from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

class analyzeSentiments:
    
    def __init__(self, user_input):
        self.user_input = user_input
        self.key_mapper = {'neu':'Neutral', 'pos':'Positive', 'neg':'Negative'}
        
    def get_sentiment_polarity(self):
        
        score_dict = SentimentIntensityAnalyzer().polarity_scores(self.user_input)
        
        # Polarity Conclusion
        compound_score = score_dict['compound']
        
        if  compound_score>= 0.05:
            polarity = 'Positive'
        elif (compound_score > -0.05) and (compound_score < 0.05):
            polarity = 'Neutral'
        else:
            polarity = 'Negative'
            
        return polarity
    
    def get_sentiment_data(self):
        
        score_dict = SentimentIntensityAnalyzer().polarity_scores(self.user_input)
        
        # Polarity Dataframe
        polarity_df = pd.DataFrame([score_dict])[['pos', 'neg', 'neu']].T.reset_index().rename(columns={'index':'Sentiment', 0:'Score'}).sort_values(by='Score', ascending=False)
        
        polarity_df['Sentiment'] = polarity_df['Sentiment'].map(self.key_mapper)
        polarity_df = polarity_df[polarity_df.Score > 0]
        
        # polarity_df = polarity_df.set_index('Sentiment')

        return polarity_df
                
        