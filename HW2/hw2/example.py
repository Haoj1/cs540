import os
import math


#These first two functions require os operations and so are completed for you
#Completed for you
def load_training_data(vocab, directory):
    """ Create the list of dictionaries """
    top_level = os.listdir(directory)
    dataset = []
    for d in top_level:
        if d[0] == '.':  # to skip hidden directories for e.g.  '.DStore' directory
            continue  #in mac which may throw errors
        if d[-1] == '/':
            label = d[:-1]
            subdir = d
        else:
            label = d
            subdir = d + "/"
        files = os.listdir(directory + subdir)
        for f in files:
            bow = create_bow(vocab, directory + subdir + f)
            dataset.append({'label': label, 'bow': bow})
    return dataset


#Completed for you
def create_vocabulary(directory, cutoff):
    """ Create a vocabulary from the training directory
        return a sorted vocabulary list
    """

    top_level = os.listdir(directory)
    vocab = {}
    for d in top_level:
        if d[0] == '.':  # to skip hidden directories for e.g.  '.DStore' directory
            continue  #in mac which may throw errors
        subdir = d if d[-1] == '/' else d + '/'
        files = os.listdir(directory + subdir)
        for f in files:
            with open(directory + subdir + f, 'r') as doc:
                for word in doc:
                    word = word.strip()
                    if not word in vocab and len(word) > 0:
                        vocab[word] = 1
                    elif len(word) > 0:
                        vocab[word] += 1
    return sorted([word for word in vocab if vocab[word] >= cutoff])


#The rest of the functions need modifications ------------------------------
#Needs modifications
def create_bow(vocab, filepath):
    """ Create a single dictionary for the data
        Note: label may be None
    """
    bow = {}

    with open(filepath, 'r') as doc:
        for word in doc:
            if word in vocab:
                if not word in bow:
                    bow[word] = 1
                else:
                    bow[word] += 1
            if not word in vocab:
                if None in bow:
                    bow[None] += 1
                else:
                    bow[None] = 1

    return bow


#Needs modifications
def prior(training_data, label_list):
    """ return the prior probability of the label in the training set
        => frequency of DOCUMENTS
    """

    smooth = 1  # smoothing factor
    logprob = {}
    # TODO: add your code here
    frequencies = {label: 0 for label in label_list}
    # frequencies['2020'] = 0
    # frequencies['2016'] = 0
    for file_data in training_data:
        if file_data['label'] in label_list:
            frequencies[file_data['label']] += 1

    logprob = {label: math.log((frequencies[label] + smooth) / (len(training_data) + smooth * len(label_list))) for label in frequencies.keys()}
    return logprob


#Needs modifications
def p_word_given_label(vocab, training_data, label):
    """ return the class conditional probability of label over all words, with smoothing """

    smooth = 1  # smoothing factor
    word_prob = {}
    # TODO: add your code here
    filtered_docs = [doc['bow'] for doc in training_data if doc['label'] == label]

    counts = {}
    for word in vocab:
        counts[word] = 0
    counts[None] = 0

    for word in vocab:
        for doc in filtered_docs:
            if word in doc:
                counts[word] += doc[word]

    for doc in filtered_docs:
        if None in doc:
            counts[None] += doc[None]

    total_words = 0
    for key in counts:
        total_words += counts[key]

    word_prob = {word: math.log((counts[word] + smooth * 1) / (total_words + smooth * (len(vocab) + 1))) for word in counts}

    return word_prob


##################################################################################
#Needs modifications
def train(training_directory, cutoff):
    """ return a dictionary formatted as follows:
            {
             'vocabulary': ,
             'log prior': ,
             'log p(w|y=2016)': ,
             'log p(w|y=2020)':
            }
    """
    retval = {}
    label_list = os.listdir(training_directory)
    # TODO: add your code here
    vocab = create_vocabulary(training_directory, cutoff)
    training_data = load_training_data(vocab, training_directory)
    log_prior = prior(training_data, label_list)  # It's better to pass ['2016', '2020'] instead of label_list
    p_word_label_2016 = p_word_given_label(vocab, training_data, '2016')
    p_word_label_2020 = p_word_given_label(vocab, training_data, '2020')

    retval['vocabulary'] = vocab
    retval['log prior'] = log_prior
    retval['log p(w|y=2016)'] = p_word_label_2016
    retval['log p(w|y=2020)'] = p_word_label_2020


    return retval


#Needs modifications
def classify(model, filepath):
    """ return a dictionary formatted as follows:
            {
             'predicted y': <'2016' or '2020'>,
             'log p(y=2016|x)': ,
             'log p(y=2020|x)':
            }
    """
    retval = {}
    # TODO: add your code here
    p_doc_2016 = model['log_prior']['2016']
    p_doc_2020 = model['log_prior']['2020']
    with open(filepath, 'r') as doc:
        for word in doc:
            word = word.strip()
            p_doc_2016 += model['log p(w|y=2016)'][word]
            p_doc_2020 += model['log p(w|y=2020)'][word]

    retval['predicted y'] = '2016' if p_doc_2016 > p_doc_2020 else '2020'
    retval['log p(y=2016|x)'] = p_doc_2016
    retval['log p(y=2020|x)'] = p_doc_2020

    return retval


if __name__ == "__main__":
    model = train('./EasyFiles/', 1)
    print(model)
    result = classify(model, './EasyFiles/2020/2.txt')
    print(result)