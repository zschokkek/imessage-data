import re
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Mapping phone numbers and 'Me' to names
NAME_MAP = {
    '+16176865741': 'Eli',
    '+15086567672': 'Nikhil',
    '+17742495038': 'Nicket',
    '+18577565980': 'Easwer',
    '+16176783154': 'Svayam',
    '+16177947953': 'Arjun',
    '+17742850915': 'Neil',
    'Me': 'Kyle'  # Mapping 'Me' to 'Kyle'
}

def parse_messages(file_contents):
    """
    Parse messages from the provided text and map phone numbers to names.
    
    Args:
        file_contents (str): The raw text data from the file.
    
    Returns:
        pd.DataFrame: A DataFrame containing parsed messages with timestamps and identifiers.
    """
    # Replace Eli's email with his phone number before parsing
    file_contents = file_contents.replace('elimendels55@gmail.com', '+16176865741')

    message_pattern = re.compile(
        r'(\w{3}\s\d{2},\s\d{4}\s\d{2}:\d{2}:\d{2}\s\w{2})\n(\+[\d]{10,}|\w+@\w+\.\w+|Me)\n(.*?)\n',
        re.DOTALL
    )
    messages = message_pattern.findall(file_contents)
    
    data = {'timestamp': [], 'identifier': [], 'message': []}
    for timestamp, identifier, message in messages:
        identifier = NAME_MAP.get(identifier, identifier)  # Map to name or keep original identifier
        data['timestamp'].append(timestamp)
        data['identifier'].append(identifier)
        data['message'].append(message.strip())
    
    return pd.DataFrame(data)

def analyze_sentiment(df):
    """
    Perform sentiment analysis on the messages and calculate average sentiment per identifier.
    
    Args:
        df (pd.DataFrame): DataFrame containing messages with identifiers.
    
    Returns:
        pd.DataFrame: A DataFrame with the average sentiment per identifier.
    """
    df['sentiment'] = df['message'].apply(lambda msg: TextBlob(msg).sentiment.polarity)
    sentiment_summary = df.groupby('identifier')['sentiment'].mean().reset_index()
    return sentiment_summary

def count_occurrences(df, term='Kyle'):
    """
    Count occurrences of a specific term (e.g., 'Kyle') in the identifier column.
    
    Args:
        df (pd.DataFrame): DataFrame containing identifiers.
        term (str): The term to count in the identifier column.
    
    Returns:
        int: The number of occurrences of the term.
    """
    return df['identifier'].value_counts().get(term, 0)

def filter_ngrams(ngrams_df):
    """
    Filter out n-grams containing three-letter words or patterns like URLs.
    
    Args:
        ngrams_df (pd.DataFrame): A DataFrame of n-grams and their counts.
    
    Returns:
        pd.DataFrame: A filtered DataFrame.
    """
    # Define a pattern to match unwanted n-grams
    filter_pattern = re.compile(r'\b\w{1,3}\b')  # Matches any word with 1 to 3 letters
    
    # Filter out n-grams containing words matching the filter pattern
    ngrams_df['is_valid'] = ngrams_df['ngram'].apply(lambda x: not any(filter_pattern.search(word) for word in x.split()))
    
    # Filter out URLs or similar patterns
    ngrams_df['is_valid'] &= ~ngrams_df['ngram'].str.contains(r'\bhttps?://|\btwitter|library|attachments|users\b', regex=True)
    
    # Return only valid n-grams
    return ngrams_df[ngrams_df['is_valid']].drop(columns=['is_valid'])

def find_common_ngrams(df, n=2, top_n=10):
    """
    Find the most common n-grams in the messages and filter out non-informative ones.
    
    Args:
        df (pd.DataFrame): DataFrame containing messages.
        n (int): The number of words in the n-gram.
        top_n (int): The number of top n-grams to return.
    
    Returns:
        pd.DataFrame: A DataFrame of the most common filtered n-grams and their counts.
    """
    vectorizer = CountVectorizer(ngram_range=(n, n), stop_words='english')
    X = vectorizer.fit_transform(df['message'])
    ngrams = vectorizer.get_feature_names_out()
    counts = X.toarray().sum(axis=0)
    ngram_counts = pd.DataFrame({'ngram': ngrams, 'count': counts})
    
    # Filter the n-grams
    filtered_ngrams = filter_ngrams(ngram_counts)
    
    return filtered_ngrams.sort_values(by='count', ascending=False).head(top_n)

def perform_lda(df, n_topics=5, n_top_words=10):
    """
    Perform Latent Dirichlet Allocation (LDA) to discover topics in the messages.
    
    Args:
        df (pd.DataFrame): DataFrame containing messages.
        n_topics (int): The number of topics to discover.
        n_top_words (int): The number of top words per topic.
    
    Returns:
        dict: A dictionary of topics and their associated top words.
    """
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['message'])
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    
    words = vectorizer.get_feature_names_out()
    topics = {
        f'Topic {idx + 1}': [words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        for idx, topic in enumerate(lda.components_)
    }
    return topics

# Main execution flow
if __name__ == "__main__":
    # Load your file content
    with open('WHAT DEY GON SAY NOW LOG.txt', 'r') as file:
        file_contents = file.read()

    # Parse the messages
    df = parse_messages(file_contents)

    # Analyze sentiment
    sentiment_summary = analyze_sentiment(df)
    print(sentiment_summary)

    # Count occurrences of 'Kyle'
    kyle_count = count_occurrences(df, term='Kyle')
    print(f"\nOccurrences of 'Kyle': {kyle_count}")

    # Find common bigrams
    bigrams = find_common_ngrams(df, n=2, top_n=10)
    print("\nCommon Bigrams:")
    print(bigrams)

    # Perform LDA for topic modeling
    topics = perform_lda(df, n_topics=5, n_top_words=10)
    print("\nDiscovered Topics:")
    for topic, words in topics.items():
        print(f"{topic}: {', '.join(words)}")