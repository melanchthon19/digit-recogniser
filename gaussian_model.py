import numpy as np
import pandas as pd
import math
import sys
from scipy.stats import multivariate_normal


class GaussianModel:
    def __init__(self, phone2features, parameters):
        # take parameters and all features
        # assign probabilities of each state producing each feature according to given parameters
        self.parameters = parameters  # dict_keys(['mean', 'sd'])
        self.phone2features = phone2features
        self.states = phone2features.keys()  # ['s', 'e', 'r', 'o', 'u', 'n', 'd', 't', 'k', 'a', 'i', 'ts', 'b']
        self.features = self.get_all_features()
        self.dimensions = len(self.features[0])  # number of feature dimension
        self.priors = self.calculate_priors()

    def get_all_features(self):
        all_features = np.array([feature for phone in self.phone2features.keys() for feature in self.phone2features[phone]])
        index2features = {index: features for index, features in enumerate(all_features)}

        return index2features  # dict_keys([0, 1, 2, 3, 4, ... N]

    def calculate_priors(self):  # the probability of randomly picking a class
        priors = {}
        total_data = len(self.features)
        for state in self.states:
            prior = len(self.phone2features[state]) / total_data  # proportion of samples initially assigned to each state
            priors[state] = prior

        total = np.sum([p for p in priors.values()])
        if total < 0.99 or total > 1.001:
            raise ValueError(f'priors for all classes (states) should sum up to 1.0\npriors sum up to: {total}')

        return priors

    def calculate_posterior(self, x):
        # p(class|x) = (p(x|class) p(class)) / p(x)
        #            =  likelihood * prior / evidence
        # p(x) acts as a scale factor to ensure that the p(x|class) for all classes sum up to one
        # because p(x) is the same for all classes, it is not computed for classification purposes

        pdfs = self.calculate_pdfs(x)  # p(x|class)
        #print(pdfs)
        posteriors = [pdfs[state] * self.priors[state] for state in self.states]
        # classification = posteriors.index(max(posteriors))
        classification = posteriors.index(posteriors)

        return classification

    def calculate_posterior_2(self, state, feature):
        print('state', state)
        print('feature', feature)


        return posterior


    def calculate_pdfs(self, x):
        log_pdfs = []
        for state in self.states:  # compute p(x|class) for all classes
            covariance = np.zeros([self.dimensions, self.dimensions])
            for row, _ in enumerate(covariance):
                covariance[row][row] = self.parameters['sd'][state][row]**2
            mean = self.parameters['mean'][state]
            print('mean',mean, '\n covariance', covariance)
            log_pdf = multivariate_normal.logpdf(x, mean=mean, cov=covariance, allow_singular=True)
            log_pdfs.append(log_pdf)
        pdfs = np.exp(log_pdfs)

        return pdfs

    def get_emission_probabilities(self):
        emission_prob = pd.DataFrame(0, index=self.states, columns=self.features.keys())
        print(emission_prob)

        # loop for each state in all states
        for state in self.states:
            # calculate posterior returns probability of the state emitting a vector feature.
            for feature in self.features.keys():
                emission_prob.loc[state, feature] = self.calculate_posterior_2(state, feature)
            break
        print(emission_prob)

