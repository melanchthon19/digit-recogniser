#!/usr/bin/env python3

import re
import os

print(os.listdir('.'))
for file in os.listdir('test2'):
    reg = re.search('\d+', file)
    number = int(reg.group()) + 70
    new_name = re.sub(reg.group(), str(number), file)
    #print(os.getcwd())
    #print(file)
    #print(new_name)
    os.rename(os.path.join('test2',file), os.path.join('test2',new_name))
