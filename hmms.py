#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os
import sys
import re
import gaussian_model as gm

# 1) uniform start
# for each phone, collect features to compute mean and sd
# each feature sequence is divided uniformly depending on number of phones (lately, number of states)
#

def read_dictionary(file_to_read):
    acoustic_dict = {}
    with open(file_to_read, 'r') as file:
        for line in file:
            word = re.match(r"'(\w+)':\s([\w+\s]+)", line)
            acoustic_dict[word.group(1)] = word.group(2).split()

    return acoustic_dict


def get_features_training(path, digits):
    spoken_digits = pd.read_csv('spoken_digits.csv', index_col=0)
    digit2index = {digit: [index+1 for index, row in spoken_digits.iterrows()
                           if row['labels'] == digit and row['set'] == 'training']
                   for digit in digits.keys()}
    #print(digit2index)

    list_of_files = os.listdir(path)
    digit2features = {digit: [] for digit in digits.keys()}
    for digit in digits.keys():
        for file in list_of_files:
            if int(re.findall(r'\d+', file)[0]) in digit2index[digit]:
                features = pd.read_csv(os.path.join(path, file), sep=';')
                features.drop(['name', 'frameTime'], axis='columns', inplace=True)
                digit2features[digit].append(features)

    return digit2features


def uniform_start(digit2features, acoustic_dict):
    phone2features = {phone: [] for digit in acoustic_dict for phone in acoustic_dict[digit]}
    #print(phone2features)
    for digit in digit2features.keys():
        for sample in range(len(digit2features[digit])):
            uniform = round(len(digit2features[digit]) / len(acoustic_dict[digit]))

            for index, phone in enumerate(acoustic_dict[digit]):
                features = digit2features[digit][sample].iloc[index:index+uniform]
                index += uniform
                phone2features[phone].append(features.values)

    for phone in phone2features.keys():
        phone2features[phone] = np.concatenate(phone2features[phone], axis=0)

    # initializing initial_parameters with standard values: mean = 0, standard deviation = 1
    initial_parameters = {'mean': {phone: 0 for phone in phone2features.keys()},
                          'sd': {phone: 1 for phone in phone2features.keys()}}
    # setting initial_parameters according to uniform segmentation (i.e. uniform start)
    for phone in phone2features.keys():
        initial_parameters['mean'][phone] = phone2features[phone].mean(axis=0)
        initial_parameters['sd'][phone] = phone2features[phone].std(axis=0)
    #for parameter in initial_parameters:
    #    for state in phone2features:
    #        print(parameter, state, initial_parameters[parameter][state].shape)

    return phone2features, initial_parameters


def transition_probabilities(acoustic_dict, phone2features):
    states = [phone for phone in phone2features.keys()]
    states.insert(0, 'start')
    states.insert(len(states), 'end')
    tran_prob = pd.DataFrame(0, index=states, columns=states)

    for word in acoustic_dict:  # assign raw counts given acoustic dictionary
        for phone in states:
            try:
                index = acoustic_dict[word].index(phone)
            except ValueError:
                continue
            i = acoustic_dict[word][index]  # phone: being in state i
            if index == 0:
                v = 'start'
            else:
                v = acoustic_dict[word][index-1]  # coming from state v
            if index == len(acoustic_dict[word])-1:
                j = 'end'
            else:
                j = acoustic_dict[word][index+1]  # going to state j
            tran_prob.loc[v, i] += 1
            tran_prob.loc[i, j] += 1
    tran_prob.loc['end', 'end'] = 1  # assigning probability once 'end' state is reached

    for phone in states:  # assigning self-loop probabilities
        if phone != 'start':
            tran_prob.loc[phone, phone] = 1

    #print(tran_prob)
    for index, row in tran_prob.iterrows():  # convert raw counts to probabilities
        total = np.sum(row.to_numpy())
        for column in states:
            count = tran_prob.loc[index, column]
            if count == 0:
                continue
            else:
                tran_prob.loc[index, column] = tran_prob.loc[index, column] / total

    for index, row in tran_prob.iterrows():  # checking distribution
        total = np.sum(row.to_numpy())
        if total > 1.0 or total < 0.99:
            raise ValueError(f'probabilities must sum up to 1.0 in row:\n{row}\ntotal: {total}')

    return tran_prob


def emission_probabilities(phone2features, parameters):
    # pass all features, initial parameters
    # according to initial parameters, compute pdf for each state for each observation
    gaussian_model = gm.GaussianModel(phone2features, parameters)
    emission_prob = gaussian_model.get_emission_probabilities()
    
    return


if __name__ == '__main__':
    path_to_features = '/Users/danielmora/git_digit_recogniser/digit-recogniser/features/egemapsv01b/'
    acoustic_dict = read_dictionary('dictionary.txt')
    print('acoustic_dict', acoustic_dict)
    digit2features = get_features_training(path_to_features, acoustic_dict)
    print('digit2features.keys()', digit2features.keys())
    phone2features, initial_parameters = uniform_start(digit2features, acoustic_dict)
    print('initial_parameters.keys()', initial_parameters.keys())
    emission_matrix = emission_probabilities(phone2features, initial_parameters)
    #print('emission_matrix', emission_matrix)
    transition_matrix = transition_probabilities(acoustic_dict, phone2features)
    #print('transition probabilities\n', transition_matrix)

    #reestimated_parameters = reestimate(phone2features, initial_parameters)
    #def reestimate(phone2features, parameters):
    #    print(phone2features['s'].shape)
        #all_features =
