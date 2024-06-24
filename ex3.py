# Neriya Shulman 208275024 Yuval Dahari 209125939

from os import path
import numpy as np
from collections import Counter

def read_lines(file_path):
    '''
    Read lines from a file and return a list of stripped lines.
    '''
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def parse_file(input_file):
    '''
    Parse the input file and return a list of wods.
    '''
    lines = read_lines(input_file)
    
    headers = [line[1:-1] for line in lines[0::2]]
    topics = [header.split()[2:] for header in headers] # extract the topics from the headers
    
    articles = [line.split() for line in lines[1::2]] # filter all the headers 
    return articles, topics

def filter_rare_words(articles):
    '''
    Filter all the words that appear less then 4 times
    '''
    words_in_develop = [word for article in articles for word in article]
    words_counter = Counter(words_in_develop)
    return [word for word, count in words_counter.items() if count > 3]
           
def create_bag_of_words(article, valid_words):
    '''
    Count the number of occurrences in the article of each word in the valid_words list
    '''
    counter = Counter(article)
    return [counter[word] for word in valid_words]

def calculate_z(px, pw_x, num_of_clusters, article_word_occurrences):
    '''
    Calculate the log(p(x) * Π p(w|x)) for each cluster
    '''
    z = np.zeros(num_of_clusters) 
    for cluster in range(num_of_clusters):
        z[cluster] = np.log(px[cluster]) + np.sum(article_word_occurrences * np.log(pw_x[cluster]))
    return z

def lidstone_smoothing(bag_of_words, lambda_, vocab_size):
    '''
    Function to calculate the Lidstone-smoothed probability of bag_of_words
    '''
    all_words = np.sum(bag_of_words)
    return (bag_of_words + lambda_) / (all_words + lambda_ * vocab_size)

def e_step(px, pw_x, word_occurrences_by_article, num_of_articles, num_of_clusters, underflow_threshold):
    '''
    Calculate px_y - the probability of topic given specific article
    '''
    px_y = np.zeros((num_of_articles, num_of_clusters))
    pyx = np.zeros((num_of_articles, num_of_clusters))

    for article in range(num_of_articles):
        z = calculate_z(px, pw_x, num_of_clusters, word_occurrences_by_article[article])
        # Normalizing by subtracting the maximum zi so the biggest value will be 0
        m = np.max(z)
        z -= m
        
        # Prevent underflow by filter very small values
        for cluster in range(num_of_clusters):
            if z[cluster] <= underflow_threshold:
                pyx[article, cluster] = 0
            else:
                pyx[article, cluster] = np.exp(z[cluster])
        px_y[article] = pyx[article] / np.sum(pyx[article])
    
    return px_y

def calculate_pw_x(num_of_clusters, num_of_words, num_of_articles, word_occurrences_by_article, px_y, lambda_, vocab_size):
    '''
    Calculate pw_x - the probability of word given specific topic
    '''
    pw_x = np.zeros((num_of_clusters, num_of_words))
    
    for cluster in range(num_of_clusters):
        articles_bow_in_cluster = np.zeros((num_of_articles, num_of_words))
        
        articles_bow_in_cluster = word_occurrences_by_article * px_y[:, cluster][:, np.newaxis] # for all y
                
        numerators = np.sum(articles_bow_in_cluster, axis=0) # Σ_t (n_tk * w_ti)
        denominators = np.sum(articles_bow_in_cluster) # Σ_t (n_t * w_ti)
        pw_x[cluster] = (numerators + lambda_) / (denominators + vocab_size * lambda_) # P_ik = P(w_k | Ci) after smoothing

    return pw_x

def m_step(px_y, word_occurrences_by_article, num_of_articles, num_of_clusters, epsilon_min_x, lambda_, vocub_size):
    '''
    Calculate the new px and pw_x
    '''
    px = np.sum(px_y, axis=0) / num_of_articles # αi = p(ci)
    px[px < epsilon_min_x] = epsilon_min_x # update the values according to threshold
    px = px / np.sum(px) # ensure the sum of the probabilities is 1
    
    num_of_words = word_occurrences_by_article.shape[1]
    pw_x = calculate_pw_x(num_of_clusters, num_of_words, num_of_articles, word_occurrences_by_article, px_y, lambda_, vocub_size)
    return px, pw_x

def calculate_log_likelihood(px, pw_x, word_occurrences_by_article, num_of_articles, num_of_clusters, underflow_threshold):
    '''
    Calculate the log likelihood - the probability of observing a set of data given a particular statistical model and its parameters
    '''
    log_likelihood = 0
    
    for article in range(num_of_articles):
        z = calculate_z(px, pw_x, num_of_clusters, word_occurrences_by_article[article])
        m = np.max(z)
        z -= m
        pyx = np.zeros(num_of_clusters)
        for cluster in range(num_of_clusters):
            pyx[cluster] = np.exp(z[cluster]) if z[cluster] > underflow_threshold else 0
            
        cur_likelihood = m + np.log(np.sum(pyx)) # ln L = m + Σ e ^ (zi - m)
        log_likelihood += cur_likelihood

    return log_likelihood

def calculate_accuracy(topic_names, articles_topics, correct_topics):
    '''
    Calculate the accuracy of the clustering according to the correct topics
    '''
    num_of_articles = len(articles_topics)
    num_of_cluster = 9
    all_num_of_success = 0
    
    for cluster in range(num_of_cluster):
        indexes = [i for i in range(len(articles_topics)) if articles_topics[i] == cluster]
        max_probs = 0
        cluster_num_of_success = 0
        if len(indexes) == 0:
            continue
        for name in topic_names:
            cur_num_of_success = len([i for i in indexes if name in correct_topics[i]])
            prop = cur_num_of_success / len(indexes)
            if prop > max_probs:
                max_probs = prop
                cluster_num_of_success = cur_num_of_success
        all_num_of_success += cluster_num_of_success
    return all_num_of_success / num_of_articles
    
def cluster_articles(articles):
    lambda_ = 0.023 # for lidstone smoothing
    epsilon = 0.01 # for the likelihood
    underflow_threshold = -10 # for the p(x|y) calculation so that e^(z) will not be underflow.
    epsilon_min_px = 0.01 # epsilon for smoothing the subject probabilities
    num_of_clusters = 9
    num_of_articles = len(articles)
    vocab_size = 300_000

    px = np.full(num_of_clusters, 1 / num_of_clusters) # uniform distribution
    
    words_in_develop = filter_rare_words(articles) 
    word_occurrences_by_article = np.array([create_bag_of_words(article, words_in_develop) for article in articles]) # bag of words for each article
    
    articles_topics = np.array([i % num_of_clusters for i in range(num_of_articles)]) # index of the article modulo 9
    
    px_y = np.zeros((num_of_articles, num_of_clusters))
    px_y[np.arange(num_of_articles), articles_topics] = 1
    pw_x = calculate_pw_x(num_of_clusters, len(words_in_develop), num_of_articles, word_occurrences_by_article, px_y, lambda_, vocab_size)
    
    likelihood = float("-inf")
    iterations = 0
    stop = False
    
    likelihoods = []
    perplexities = []

    while not stop:
        px_y = e_step(px, pw_x, word_occurrences_by_article, num_of_articles, num_of_clusters, underflow_threshold) # w_ti in the document
        px, pw_x = m_step(px_y, word_occurrences_by_article, num_of_articles, num_of_clusters, epsilon_min_px, lambda_, vocab_size)
        
        new_likelihood = calculate_log_likelihood(px, pw_x, word_occurrences_by_article, num_of_articles, num_of_clusters, underflow_threshold)

        iterations += 1
        print(f'Iteration {iterations}\tlikelihood: {new_likelihood}')
        
        if new_likelihood - likelihood < epsilon:
            break
        likelihood = new_likelihood
        likelihoods.append(likelihood)

        perplexity = np.exp(-likelihood / np.sum(word_occurrences_by_article)) # 2 ^ (-log(likelihood) / N)
        perplexities.append(perplexity) 
        
    return np.argmax(px_y, axis=1), likelihood

if __name__ == "__main__":
    base_path = path.dirname(__file__)
    dev_input = path.join(base_path, "develop.txt")
    articles, correct_topics = parse_file(dev_input)
    topic_names = read_lines(path.join(base_path, "topics.txt"))

    articles_topics, likelihood = cluster_articles(articles)
    accuracy = calculate_accuracy(topic_names, articles_topics, correct_topics)
    print("-----------------------------------------")
    print(f'Accuracy: {accuracy}')
