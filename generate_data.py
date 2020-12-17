import random
import pandas as pd


def generate_digits(n):
    '''
    function that returns a list of numbers from 0 to 9 randomly ordered repeated n times
    :param n: integer (amount) of times that the sequence 0-9 are to be retrieved
    :return: a list of numbers
    for n == 2 --> [9, 1, 0, 4, 6, 5, 2, 7, 8, 3, 3, 4, 0, 1, 9, 6, 5, 7, 8, 2]
    '''
    digits = [random.sample(range(10), 10) for i in range(n)]
    digits = [d for digit in digits for d in digit]

    return digits


def generate_labels(spoken_digits, labels):
    '''
    function that generates a dataframe to keep track of the spoken digits generated
    :param spoken_digits: list of recorded digits
    :param labels: dictionary with corresponding labels for each digit
    :return: dataframe with data labelled
    '''

    data = pd.DataFrame({'digits':[digit for digit in spoken_digits],
                         'labels':[labels[digit] for digit in spoken_digits]})

    return data


def split_data(data, training_percentage):
    '''
    function that assigns training or testing label to each sample
    :param data: dataframe with digits and labels
    :param training: float between 0 and 1 indicating the percentage samples that will belong to training set
    :return: dataframe with set label added as new column
    '''
    total_samples = len(data.index)
    sets = ['training'] * int(total_samples*training_percentage) + ['testing'] * int(total_samples*(1-training_percentage))
    data['set'] = sets

    return data

labels = {0:'cero', 1:'uno', 2:'dos', 3:'tres', 4:'cuatro',
          5:'cinco', 6:'seis', 7:'siete', 8:'ocho', 9:'nueve'}

spoken_digits = generate_digits(10)
data = generate_labels(spoken_digits, labels)
data = split_data(data, training_percentage=0.7)

# to store data in csv file
#data.to_csv('spoken_digits.csv')