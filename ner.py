import spacy

nlp = spacy.load("en_core_web_trf")

def perform_ner_on_summary(summary):
    doc = nlp(summary)
    entities = []
    for ent in doc.ents:
        entities.append((ent.text, ent.label_))
    return entities

summary = "Obama visited New York City to address climate change issues."
extracted_entities = perform_ner_on_summary(summary)
for entity, label in extracted_entities:
    print(f"Entity: {entity}, Label: {label}")
