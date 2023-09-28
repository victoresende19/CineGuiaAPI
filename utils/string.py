import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")

# Diretor
def name_treat(nome):
    if isinstance(nome, list):
        return [str.lower(i.replace(" ", "")) for i in nome]
    else:
        if isinstance(nome, str):
            return str.lower(nome.replace(" ", ""))
        else:
            return ''

# Taglines
def remover_pontuacao(texto):
    return texto.translate(texto.maketrans('', '', string.punctuation))

def tagline_treat(texto, stopwords=stopwords.words('english')):
    try:
        texto = remover_pontuacao(texto)
        return [i.lower() for i in texto.split() if i.lower() not in stopwords]
    except:
        return []


def gruped_metadados(x):
    return '' + ' '.join(x['director_lst']) + ' ' + ' '.join(x['cast_lst']) + ' ' + ' '.join(x['genres_lst']) + ' ' + ' '.join(x['keywords_lst'])
