# Digit Recogniser

(in progress)

Basic Digit Recogniser meant to go step by step in the process from data generation to training and evaluation. Common library is used to explicitly go through each step.

#### generate_data.py
It randomly generates a list of numbers from 0 to 9 and outputs a csv file with appropiate labels. It also splits all the data into training (70%) and test (%30) sets.

#### extract_features.py
Once digits are recorded and stored in the folder ```wav```, then features are extracted using openSMILE toolkit.
Using openSMILE different sets of features can be extracted by specifying the .conf when calling openSMILE.\
It outputs a csv file containing features per digit sample that are stored in the folder ```features```.\
When using openSMILE, remember to change frameModeFunctionals.conf.inc. By degault, the frameMode is 'full' meaning that it will take the average across the entire audio. However, we don't want just one value, but features extracted according a specific window size and every specific step. In this case, I used a window size of 25 mm and 10 mm step.

```
frameMode = fixed
frameSize = 0.25
frameStep = 0.10
frameCenterSpecial = lef
```
\
TODO:
- fix reestimated emission matrix in hmms.py
