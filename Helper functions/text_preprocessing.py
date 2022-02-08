import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re
nltk.download('all')
# define function to preprocess text
def lemmatization(sentence):
    lemm_obj=WordNetLemmatizer()
    sentence=re.sub('[^A-Za-z]',' ',sentence)
    sentence=re.sub('[\s]+',' ',sentence)
    sentence=sentence.lower()
    sentence=sentence.split()
    # sentence=include_pos(sentence)
    # sentence=[lemm_obj.lemmatize(word,pos_tag) for word,pos_tag in sentence]
    sentence=[lemm_obj.lemmatize(word) for word in sentence]
    sentence=" ".join(sentence)
    return sentence


def lemmatization_with_pos(sentence):
    lemm_obj=WordNetLemmatizer()
    sentence=re.sub('[^A-Za-z]',' ',sentence)
    sentence=re.sub('[\s]+',' ',sentence)
    sentence=sentence.lower()
    sentence=sentence.split()
    sentence=include_pos(sentence)
    lemmatized_sentence = []
    for word, tag in sentence:
        if tag is None:
            # if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:	
            # else use the tag to lemmatize the token
            lemmatized_sentence.append(lemm_obj.lemmatize(word, tag))
    sentence = " ".join(lemmatized_sentence)
    return sentence

def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def include_pos(words):
    
    '''accepts words as input and assigns it the correct part of speech in context to the sentence'''
    pos_tagged = nltk.pos_tag(words)
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    return wordnet_tagged
    
    

    
    