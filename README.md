# Digit Recogniser

(in progress)

Basic Digit Recogniser meant to go step by step in the process from data generation to training and evaluation. Common library is used to explicitly go through each step.

When using openSMILE, remember to change frameModeFunctionals.conf.inc. By degault, the frameMode is 'full' meaning that it will take the average across the entire audio. However, we don't want just one value, but features extracted according a specific window size and every specific step. In this case, I used a window size of 25 mm and 10 mm step.

```
frameMode = fixed
frameSize = 0.25
frameStep = 0.10
frameCenterSpecial = lef
```
