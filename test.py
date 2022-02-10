# streamlit_app.py
import spacy_streamlit
from spacy_streamlit import visualize_parser, visualize_ner
import spacy 

# models = ["en_core_web_sm", "en_core_web_md"]
default_text = "Amazing experience, even when buying from home due to covid! James Eadington in particular was a fantastic salesman. Many happy miles with the 718 cayman GTS!"
# spacy_streamlit.visualize(models, default_text)
nlp = spacy.load("en_core_web_sm")
doc = nlp(default_text)
visualize_ner(doc, labels=nlp.get_pipe("ner").labels, show_table=False)