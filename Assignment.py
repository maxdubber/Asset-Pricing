
# coding: utf-8

# # Part A.1


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, style
style.use('ggplot')
import statsmodels.api as sm




df = pd.read_csv('Average Value Weighted Returns -- Monthly.csv', index_col = 0);
df.index = pd.to_datetime(df.date, format='%Y%m')
df = df.drop('date', axis = 1)


mu = np.matrix(df.mean()).T
SigmaInv = np.linalg.inv(df.cov())
iota = np.matrix(np.ones(mu.shape))

A = mu.T*SigmaInv*mu
B = mu.T*SigmaInv*iota
C = iota.T*SigmaInv*iota

pi1 = SigmaInv*iota/C
pim = SigmaInv*mu/B

f = lambda mu: float(np.sqrt((A-2*B*mu+C*mu**2)/(A*C-B*B)))


mu = [i for i in np.arange(0,4.5,.01)]
sigma = [f(m) for m in mu]
#sigma
plt.plot(sigma, mu);
plt.scatter(df.std(), df.mean());
plt.scatter([np.sqrt(A/B**2), np.sqrt(1/C)], [A/B, B/C], color='k');


# In[ ]:



