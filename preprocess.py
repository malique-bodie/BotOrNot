from nltk import tokenize
from nltk.util import pad_sequence
import tensorflow as tf
import numpy as np
import pandas as pd
import torch
from nltk.tokenize import TweetTokenizer

def clean_df(df):
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['updated'] = pd.to_datetime(df['updated'])
    df['has_location'] = df['location'].apply(lambda x : 0 if x==x else 1)
    df['has_avatar'] = df['default_profile_image'].apply(lambda x : 1 if x==x else 0)
    df['has_background'] = df['profile_use_background_image'].apply(lambda x : 1 if x==x else 0)
    df['is_verified'] = df['verified'].apply(lambda x : 1 if x==x else 0)
    df['is_protected'] = df['protected'].apply(lambda x : 1 if x==x else 0)
    df['profile_modified'] = df['default_profile'].apply(lambda x : 0 if x==x else 1)
    df = df.rename(index=str, columns={'screen_name': 'username', 'statuses_count': 'total_tweets', 'friends_count': 'total_following', 'followers_count': 'total_followers', 'favourites_count': 'total_likes'})
    return df[['username', 'has_location', 'is_verified', 'total_tweets', 'total_following', 'total_followers', 'total_likes', 'has_avatar', 'has_background', 'is_protected', 'profile_modified']]


def get_user_data():
    # Read in bot and real user dataframes
    bot_users = pd.concat([pd.read_csv('user_data/social_spambots_1.csv'), pd.read_csv('user_data/social_spambots_2.csv'), pd.read_csv('user_data/social_spambots_3.csv')]).reset_index(drop=True)
    real_users = pd.read_csv('user_data/geniune_accounts.csv')

    # Define needed user properties
    cols = ['screen_name', 'created_at', 'updated', 'location', 'verified', 'statuses_count', 'friends_count', 'followers_count', 'favourites_count', 'default_profile_image', 'profile_use_background_image', 'protected', 'default_profile']
    bot_users = bot_users[cols]
    real_users = real_users[cols]

    #Clean Data
    bot_users = clean_df(bot_users)
    real_users = clean_df(real_users)

    # Add classification labels
    bot_users['Bot'] = 1
    real_users['Bot'] = 0

    # Combine dataframes
    all_users = pd.concat([bot_users, real_users])

    # Shuffle dataframe
    final = all_users.sample(frac=1).reset_index(drop=True)

    # Split into Train and Test
    train = final.drop('username', axis=1)[:int(all_users.shape[0] * 0.8)]
    test = final.drop('username', axis=1)[int(all_users.shape[0] * 0.8):]

    # Standardize data
    standardize = [ 'total_tweets', 'total_following', 'total_followers', 'total_likes']
    training_mean = train[standardize].mean()
    training_std = train[standardize].std()

    train[standardize] = (train[standardize] - training_mean)/training_std
    test[standardize] = (test[standardize] - training_mean)/training_std

    # Split into features and labels
    X_train = train.drop(['Bot', 'is_protected'], axis=1).values
    y_train = train['Bot'].values.reshape(-1,1)

    X_test = test.drop(['Bot', 'is_protected'], axis=1).values
    y_test = test['Bot'].values.reshape(-1,1)

    return X_train, y_train, X_test, y_test








def load_embeddings():
    words = []
    idx = 0
    word2idx = {}
    vectors = []
    with open('glove/glove.twitter.27B.200d.txt', 'rb') as f:
        for l in f:
            line = l.decode().split()
            w = line[0]
            words.append(w)
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)
            word2idx[w] = idx

        glove = {w: vectors[word2idx[w]] for w in words}

    return glove


def tokenize(sentence, vocab):
    tokens = []
    tweet_tokenizer = TweetTokenizer()
    for word in tweet_tokenizer.tokenize(sentence):
        if word.lower() not in vocab:
            vocab[word.lower()] = 1
        tokens.append(word.lower())
    return tokens

def get_weights(glove, vocab):
    matrix_len = len(vocab.keys())
    weights_matrix = np.zeros((matrix_len + 2, 200))
    words_found = 0

    for i, word in enumerate(vocab.keys()):
        if word in glove: 
            weights_matrix[i+2] = glove[word]
            words_found += 1
            vocab[word] = i
        else:
            vocab[word] = 0
    weights_matrix[0] = np.zeros((200)) # the <UNK>
    weights_matrix[1] = np.zeros((200)) # the <PAD>

    print(weights_matrix)
    return weights_matrix


# handles padding and unk
def word_to_index(arr, vocab):
    arr_id = []
    for s in arr:
        arr_id.append([vocab[str(word)] for word in s[0]])

    
    return tf.keras.preprocessing.sequence.pad_sequences(arr_id, maxlen=20, padding="post", value=1)

    




def get_tweet_data():
    vocab = {}
    # Read in bot and real user dataframes
    # bot_tweets = pd.concat([pd.read_csv('tweet_data/social_spambots_1.csv',dtype={'text': str}, encoding='latin-1'), pd.read_csv('tweet_data/social_spambots_2.csv',dtype={'text': str}, encoding='latin-1'), pd.read_csv('tweet_data/social_spambots_3.csv', dtype={'text': str}, encoding='latin-1')]).reset_index(drop=True)
    bot_tweets = pd.read_csv('tweet_data/social_spambots_2.csv',dtype={'text': str}, encoding='latin-1')
    real_tweets = pd.read_csv('tweet_data/geniune_accounts.csv', dtype={'text': str}, encoding='latin-1')

    # Isolate text
    bot_tweets = bot_tweets[['text']].astype(str)
    real_tweets = real_tweets[['text']].astype(str)


    # Add classification labels
    bot_tweets['Bot'] = 1
    real_tweets['Bot'] = 0

    # Combine dataframes
    all_tweets = pd.concat([bot_tweets, real_tweets])

    # Tokenize and Pad Data
    all_tweets['text'] = all_tweets['text'].apply(lambda x : tokenize(x, vocab))

    # Get Vocab
    print('getting word embeddings')

    glove = load_embeddings()

    # make weights matrix
    weights = get_weights(glove, vocab)

    # Shuffle dataframe
    final = all_tweets.sample(frac=1).reset_index(drop=True)

    # Split into Train and Test
    train = final[:int(all_tweets.shape[0] * 0.8)]
    test = final[int(all_tweets.shape[0] * 0.8):]

    # Split into features and labels
    X_train = train.drop(['Bot'], axis=1).values
    y_train = train['Bot'].values.reshape(-1,1)

    X_test = test.drop(['Bot'], axis=1).values
    y_test = test['Bot'].values.reshape(-1,1)


    return word_to_index(X_train,vocab), y_train, word_to_index(X_test, vocab), y_test, weights


