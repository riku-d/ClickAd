from keybert import KeyBERT

kw_model = KeyBERT()

def extract_keywords(text, num_keywords=5):
    keywords = kw_model.extract_keywords(text, stop_words='english', top_n=num_keywords)
    return [kw for kw, _ in keywords]
