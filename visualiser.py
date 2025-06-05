from spacy import load
from spacy import displacy
import os
from spacy.tokens import DocBin

nlp = load("es_core_news_md")

# change this to suit needs!
db = DocBin().from_disk("./testing.spacy")

# Convert the DocBin object to a list of Doc objects
docs = list(db.get_docs(nlp.vocab))

# Iterate over the Doc objects and visualize them using displacy
displacy.serve(docs[2], style="ent", auto_select_port=True, host="localhost")
