#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 18:50:11 2019

@author: manoj
"""


from densenet import DenseNet2D
model_dict = {}

model_dict['densenet'] = DenseNet2D(dropout=True,prob=0.2)
