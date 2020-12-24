import numpy as np
import pandas as pd
import math
import sys
import scipy.stats # multivariate_normal


class GaussianModel:
    def __init__(self, phone2features, index2features, parameters):
        # take parameters and all features
        # assign probabilities of each state producing each feature according to given parameters
        self.parameters = parameters  # dict_keys(['mean', 'sd'])
        self.phone2features = phone2features
        self.states = phone2features.keys()  # ['s', 'e', 'r', 'o', 'u', 'n', 'd', 't', 'k', 'a', 'i', 'ts', 'b']
        self.index2features = index2features
        self.dimensions = len(self.index2features[0])  # number of feature dimension (e.g. 88)
        self.priors = self.calculate_priors()


    def calculate_priors(self):  # the probability of randomly picking a class
        priors = {}
        total_data = len(self.index2features)
        for state in self.states:
            prior = len(self.phone2features[state]) / total_data  # proportion of samples initially assigned to each state
            priors[state] = prior

        total = np.sum([p for p in priors.values()])
        if total < 0.99 or total > 1.001:
            raise ValueError(f'priors for all classes (states) should sum up to 1.0\npriors sum up to: {total}')

        return priors


    def get_emission_probabilities(self):
        emission_prob = pd.DataFrame(0, index=self.states, columns=self.index2features.keys())
        for index in self.index2features.keys():
            for state in self.states:
                posterior = self.calculate_posterior(state, self.index2features[index])
                emission_prob.loc[state, index] = posterior  # np.exp(posterior)
                #break
            #break
        #emission_prob['sum'] = emission_prob.sum(axis=1)

        # emission_prob is a dataframe with log probs of each state emitting each feature vector

        return emission_prob


    def calculate_posterior(self, state, x):
        # p(class|x) = (p(x|class) p(class)) / p(x)
        #            =  likelihood * prior / evidence
        # p(x) acts as a scale factor to ensure that the p(x|class) for all classes sum up to one
        # because p(x) is the same for all classes, it is not computed for classification purposes

        log_pdf = self.calculate_logpdf(state, x)  # p(x|class)
        posterior = log_pdf + np.log(self.priors[state])

        return posterior


    def calculate_logpdf(self, state, x):
        covariance = np.identity(self.dimensions)  # np.zeros([self.dimensions, self.dimensions])
        for index in range(self.dimensions):
            covariance[index][index] = self.parameters['sd'][state][index]**2
        mean = self.parameters['mean'][state]

        #  allow_singular = True (?)
        log_pdf = scipy.stats.multivariate_normal.logpdf(x, mean=mean, cov=covariance, allow_singular=True)

        return log_pdf
