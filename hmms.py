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
class HMM:
    def __init__(self, path_to_features, file_to_read):
        self.acoustic_dict = self.read_dictionary(file_to_read)

        self.digit2features = self.get_features_training(path_to_features)
        self.phone2features, self.parameters = self.uniform_start()
        self.states = self.phone2features.keys()
        self.index2features = self.get_all_features()

        self.emission_matrix = self.emission_probabilities(self.parameters)
        self.transition_matrix = self.transition_probabilities()

        print('HMM object initialized with flat start')


    def read_dictionary(self, file_to_read):
        acoustic_dict = {}
        with open(file_to_read, 'r') as file:
            for line in file:
                word = re.match(r"'(\w+)':\s([\w+\s]+)", line)
                acoustic_dict[word.group(1)] = word.group(2).split()

        return acoustic_dict


    def get_all_features(self):
        all_features = np.array([feature for phone in self.phone2features.keys() for feature in self.phone2features[phone]])
        index2features = {index: features for index, features in enumerate(all_features)}

        return index2features  # dict_keys([0, 1, 2, 3, 4, ... N]


    def get_features_training(self, path_to_features):
        spoken_digits = pd.read_csv('spoken_digits.csv', index_col=0)
        digit2index = {digit: [index+1 for index, row in spoken_digits.iterrows()
                               if row['labels'] == digit and row['set'] == 'training']
                       for digit in self.acoustic_dict.keys()}
        #print(digit2index)

        list_of_files = os.listdir(path_to_features)
        digit2features = {digit: [] for digit in self.acoustic_dict.keys()}
        for digit in self.acoustic_dict.keys():
            for file in list_of_files:
                if int(re.findall(r'\d+', file)[0]) in digit2index[digit]:
                    features = pd.read_csv(os.path.join(path_to_features, file), sep=';')
                    features.drop(['name', 'frameTime'], axis='columns', inplace=True)
                    digit2features[digit].append(features)

        return digit2features


    def uniform_start(self):
        phone2features = {phone: [] for digit in self.acoustic_dict for phone in self.acoustic_dict[digit]}
        #print(phone2features)
        for digit in self.digit2features.keys():
            for sample in range(len(self.digit2features[digit])):
                uniform = round(len(self.digit2features[digit]) / len(self.acoustic_dict[digit]))

                for index, phone in enumerate(self.acoustic_dict[digit]):
                    features = self.digit2features[digit][sample].iloc[index:index+uniform]
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


    def equal_parameters(self, old_parameters, new_parameters):
        # returns True if old and new parameters are equal
        # this means that parameters were not updated,
        # thus the iteration converged
        for parameter in old_parameters.keys():
            for state in old_parameters[parameter]:
                for index in range(len(old_parameters[parameter][state])):
                    if old_parameters[parameter][state][index] != new_parameters[parameter][state][index]:
                        return False
        return True


    def reestimate_parameters(self):
        converged = False
        # reestimating parameters following soft alignment (weighted by its probability)
        # multiply each feature vector by the probability of the given state emitting that feature vector.
        # traverse states. per each state, traverse feature vectors.
        # per each vector, multiply by the probability of emitting that vector.
        # sum up this result for all features and based on that compute the mean and standard deviation.
        i = 0
        while not converged:
            print(f'reestimating parameters: iteration {i}')
            reestimated_parameters = {'mean':{phone: 0 for phone in self.phone2features.keys()},
                                      'sd':{phone: 1 for phone in self.phone2features.keys()}}

            for state in self.states:
                new_parameter = []
                for index in self.index2features:
                    feature = self.index2features[index]
                    weight = np.exp(self.emission_matrix.loc[state, index])  # np.exp to revert from log to probability
                    if weight == 0:
                        weight = 0.000000001
                    weighted_feature = feature * weight
                    new_parameter.append(weighted_feature)
                new_parameter = np.array(new_parameter)
                reestimated_parameters['mean'][state] = new_parameter.mean(axis=0)
                reestimated_parameters['sd'][state] = new_parameter.std(axis=0)

            reestimated_emission_matrix = self.emission_probabilities(reestimated_parameters)

            converged = self.equal_parameters(self.parameters, reestimated_parameters)

            if not converged:
                i += 1
                self.parameters = reestimated_parameters
                self.emission_matrix = reestimated_emission_matrix

        return reestimated_parameters, reestimated_emission_matrix


    def transition_probabilities(self):
        states = [phone for phone in self.phone2features.keys()]
        states.insert(0, 'start')
        states.insert(len(states), 'end')
        tran_prob = pd.DataFrame(0, index=states, columns=states)

        for word in self.acoustic_dict:  # assign raw counts given acoustic dictionary
            for phone in states:
                try:
                    index = self.acoustic_dict[word].index(phone)
                except ValueError:
                    continue
                i = self.acoustic_dict[word][index]  # phone: being in state i
                if index == 0:
                    v = 'start'
                else:
                    v = self.acoustic_dict[word][index-1]  # coming from state v
                if index == len(self.acoustic_dict[word])-1:
                    j = 'end'
                else:
                    j = self.acoustic_dict[word][index+1]  # going to state j
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
                    tran_prob.loc[index, column] = np.log(0.000000001)
                    #continue
                else:
                    tran_prob.loc[index, column] = np.log(tran_prob.loc[index, column] / total)

        for index, row in tran_prob.iterrows():  # checking distribution
            total = np.sum(np.exp(row.to_numpy()))
            if total > 1.001 or total < 0.99:
                raise ValueError(f'probabilities must sum up to 1.0 in row:\n{row}\ntotal: {total}')

        return tran_prob


    def emission_probabilities(self, parameters):
        # pass all features, initial parameters
        # according to initial parameters, compute pdf for each state for each observation
        gaussian_model = gm.GaussianModel(self.phone2features, self.index2features, parameters)
        emission_prob = gaussian_model.get_emission_probabilities()

        return emission_prob



if __name__ == '__main__':
    path_to_features = '/Users/danielmora/git_digit_recogniser/digit-recogniser/features/egemapsv01b/'
    hmm = HMM(file_to_read='dictionary.txt', path_to_features=path_to_features)
    print('acoustic_dict', hmm.acoustic_dict)
    print('digit2features.keys()', hmm.digit2features.keys())
    print('parameters.keys()', hmm.parameters.keys())
    print('emission matrix', hmm.emission_matrix)
    print('transition matrix\n', hmm.transition_matrix)

    hmm.parameters, hmm.emission_matrix = hmm.reestimate_parameters()
    print('emission matrix reestimated\n', hmm.emission_matrix)
