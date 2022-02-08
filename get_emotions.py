import text2emotion as te

class Emotions:
    
    def __init__(self, user_input):
        self.user_input = user_input
    
    def extract_emotions(self):
        
        ''' Function to Extract Emotions '''
        
        emotion_dict = te.get_emotion(self.user_input)
        
        return emotion_dict