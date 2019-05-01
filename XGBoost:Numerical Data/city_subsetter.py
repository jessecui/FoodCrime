#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 21:49:38 2019

@author: jessecui
"""

import pandas as pd

df = pd.read_csv('data_finalized_with_city.csv')
city_df = df.loc[df['city'] == 'Charlotte']
city_df.to_csv('Charlotte_data_full.csv')