# streamlit_app.py
import spacy_streamlit
from spacy_streamlit import visualize_parser, visualize_ner
import spacy 

# models = ["en_core_web_sm", "en_core_web_md"]
default_text = "Google was founded in September 1998 by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University in California. Together they own about 14 percent of its shares and control 56 percent of the stockholder voting power through supervoting stock. They incorporated Google as a California privately held company on September 4, 1998, in California. Google was then reincorporated in Delaware on October 22, 2002."
# spacy_streamlit.visualize(models, default_text)
nlp = spacy.load("en_core_web_sm")
doc = nlp(default_text)
visualize_ner(
                doc, 
                labels=["PERSON", "DATE", "GPE"], 
                show_table=False,
                title="Persons, dates and locations"
            )
visualize_tokens(doc, attrs=["text", "pos_", "dep_", "ent_type_"])