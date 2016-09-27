# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 12:35:24 2016

@author: maxdu
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, style
style.use('ggplot')
from statsmodels import api as sm
from scipy import stats
import pyperclip

betas = pd.read_excel('betas.xlsx')

plt.plot(betas, c='b')
plt.xlabel('Date')
plt.ylabel('Slope coefficient')

mean = np.mean(betas)
sample_std = np.std(betas)
stats.t.cdf(-mean/sample_std, 612)

pyperclip.copy(betas.describe().round(3).to_latex())
betas.skew()
betas.kurt()
betas.hist(bins=20)
stats.jarque_bera(betas)