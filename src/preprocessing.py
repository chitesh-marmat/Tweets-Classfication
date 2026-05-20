import re
import nltk
import emoji
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag


def download_nltk_resources():
    for resource in ["stopwords", "punkt", "punkt_tab", "wordnet", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
        nltk.download(resource, quiet=True)


def get_wordnet_pos(word: str) -> str:
    """Map NLTK POS tag to a WordNet POS so lemmatization is more accurate."""
    tag = pos_tag([word])[0][1][0].upper()
    tag_map = {"J": wordnet.ADJ, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_map.get(tag, wordnet.NOUN)


def clean_text(text: str) -> str:
    """
    Clean a single tweet:
    - strip URLs, @mentions, hashtag symbols, emojis
    - lowercase and remove non-alpha chars
    - tokenize, remove stopwords, lemmatize
    """
    if not isinstance(text, str):
        return ""

    # remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # remove @mentions and # symbols
    text = re.sub(r"@\w+", "", text)
    text = text.replace("#", "")
    # strip emojis
    text = emoji.replace_emoji(text, replace="")
    # lowercase and keep only letters
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    cleaned = [
        lemmatizer.lemmatize(t, get_wordnet_pos(t))
        for t in tokens
        if t not in stop_words and len(t) > 1
    ]

    return " ".join(cleaned)


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Apply cleaning to a DataFrame. Fills NaN keyword/location with empty string."""
    df = df.copy()
    df[text_col] = df[text_col].apply(clean_text)
    if "keyword" in df.columns:
        df["keyword"] = df["keyword"].fillna("")
    if "location" in df.columns:
        df["location"] = df["location"].fillna("")
    return df
