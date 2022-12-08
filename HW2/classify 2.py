import os
import math

#These first two functions require os operations and so are completed for you
#Completed for you
def load_training_data(vocab, directory):
    """ Create the list of dictionaries """
    top_level = os.listdir(directory)
    dataset = []
    for d in top_level:
        if d.startswith('.'):
            # ignore hidden files
            continue
        if d[-1] == '/':
            label = d[:-1]
            subdir = d
        else:
            label = d
            subdir = d+"/"
        files = os.listdir(directory+subdir)
        for f in files:
            bow = create_bow(vocab, directory+subdir+f)
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
        if d.startswith('.'):
            # ignore hidden files
            continue
        subdir = d if d[-1] == '/' else d+'/'
        files = os.listdir(directory+subdir)
        for f in files:
            with open(directory+subdir+f,'r') as doc:
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
    # TODO: add your code here
    file = open(filepath)
    for word in file:
        word = word.strip()
        if word in vocab:
            if word in bow:
                bow[word] += 1
            else:
                bow[word] = 1
        else:
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

    smooth = 1 # smoothing factor
    logprob = {}
    # TODO: add your code here
    counts = {label: 0 for label in label_list}
    for data in training_data:
        counts[data['label']] += 1

    possibility = [(counts[label] + smooth) / (len(training_data) + \
                   smooth * len(label_list)) for label in label_list]

    for i in range(len(label_list)):
        logprob[label_list[i]] = math.log(possibility[i])

    return logprob

#Needs modifications
def p_word_given_label(vocab, training_data, label):
    """ return the class conditional probability of label over all words, with smoothing """

    smooth = 1 # smoothing factor
    word_prob = {}
    # TODO: add your code here
    total_counts = 0
    counts = {word: 0 for word in vocab}
    for word in vocab:
        for file in training_data:
            if file['label'] == label:
                if word in file['bow']:
                    counts[word] += file['bow'][word]

    counts[None] = 0
    for file in training_data:
        if file['label'] == label and None in file['bow']:
            counts[None] += file['bow'][None]

    for key in counts:
        total_counts += counts[key]

    for word in counts:
        orig_possibility = (counts[word] + smooth * 1) / (total_counts + smooth * (len(vocab) + 1))
        word_prob[word] = math.log(orig_possibility)
    return word_prob


##################################################################################
#Needs modifications
def train(training_directory, cutoff):
    """ return a dictionary formatted as follows:
            {
             'vocabulary': <the training set vocabulary>,
             'log prior': <the output of prior()>,
             'log p(w|y=2016)': <the output of p_word_given_label() for 2016>,
             'log p(w|y=2020)': <the output of p_word_given_label() for 2020>
            }
    """
    retval = {}
    label_list = [f for f in os.listdir(training_directory) if not f.startswith('.')] # ignore hidden files
    # TODO: add your code here
    vocabulary = create_vocabulary(training_directory, cutoff)
    training_data = load_training_data(vocabulary, training_directory)
    doc_prior = prior(training_data, label_list)

    p_2016 = p_word_given_label(vocabulary, training_data, '2016')
    p_2020 = p_word_given_label(vocabulary, training_data, '2020')

    retval['vocabulary'] = vocabulary
    retval['log prior'] = doc_prior
    retval['log p(w|y=2016)'] = p_2016
    retval['log p(w|y=2020)'] = p_2020

    return retval

#Needs modifications
def classify(model, filepath):
    """ return a dictionary formatted as follows:
            {
             'predicted y': <'2016' or '2020'>,
             'log p(y=2016|x)': <log probability of 2016 label for the document>,
             'log p(y=2020|x)': <log probability of 2020 label for the document>
            }
    """
    retval = {}
    # TODO: add your code here
    file = open(filepath)
    #possibility for each year
    p_2016 = model['log prior']['2016']
    p_2020 = model['log prior']['2020']

    for word in file:
        word = word.strip()
        if word in model['log p(w|y=2016)']:
            p_2016 += model['log p(w|y=2016)'][word]
        else:
            p_2016 += model['log p(w|y=2016)'][None]
        if word in model['log p(w|y=2020)']:
            p_2020 += model['log p(w|y=2020)'][word]
        else:
            p_2020 += model['log p(w|y=2020)'][None]

    retval['log p(y=2020|x)'] = p_2020
    retval['log p(y=2016|x)'] = p_2016
    if p_2016 > p_2020:
        retval['predicted y'] = '2016'
    else:
        retval['predicted y'] = '2020'

    return retval


# if __name__ == "__main__":
#     model = train('./corpus/training/', 2)
#     result = classify(model, './corpus/test/2016/0.txt')
#     print(result)
