
# coding: utf-8

# # Part A.1


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, style
style.use('ggplot')
import statsmodels.api as sm
from scipy import stats
import pyperclip


plt.close('all')

X = pd.read_csv('Average Value Weighted Returns -- Monthly.csv', index_col = 0);
Y = pd.read_csv('market_data.csv', index_col = 0)
T,n = X.shape
#market portfolio
market_returns = Y['Mkt-RF']+Y['RF']
mu_market = market_returns.mean()
vol_market = market_returns.std()

X['market'] = market_returns
#%% 2
X.describe()
X.corr()

#%% 3
mu = np.matrix(X.mean()).T
SigmaInv = np.linalg.inv(X.cov())
iota = np.matrix(np.ones(mu.shape))

A = mu.T*SigmaInv*mu
B = mu.T*SigmaInv*iota
C = iota.T*SigmaInv*iota

f = lambda mu: float(np.sqrt((A-2*B*mu+C*mu**2)/(A*C-B*B)))

MU = [i for i in np.arange(0,4.5,.01)]
SIGMA = [f(m) for m in MU]

plt.plot(SIGMA, MU);
plt.scatter(X.std(), X.mean());
plt.scatter([np.sqrt(A/B**2), np.sqrt(1/C)], [A/B, B/C], color='k');
plt.scatter(vol_market, mu_market, s=500, c='r', marker='*')

# global minimum variance portfolio and 'the other portfolio'
pi1 = SigmaInv*iota/C #global minimum variance
pim = SigmaInv*mu/B #other

#%% 4
# make X excess return
X_e = X.subtract( Y['RF'], axis = 0)

mu_e = np.matrix(X_e.mean()).T
rf = Y['RF'].mean() #take the average of the riskfree rate as the riskfree rate for the next period
SigmaInv_e = np.linalg.inv(X_e.cov())

#tangency portfolio
pi_star = (SigmaInv_e*mu_e)/(iota.T*SigmaInv_e*mu_e)  #weights
mu_tangency = float(np.matrix(X.mean())*pi_star) #expected return
mu_e_tangency = float(np.matrix(X_e.mean())*pi_star)
mu_e_market = X_e['market'].mean()
vol_tangency = float(np.sqrt(pi_star.T*np.matrix(X.cov())*pi_star)) #volatility
vol_e_tangency = float(np.sqrt(pi_star.T*np.matrix(X_e.cov())*pi_star)) #volatility
vol_e_market = X_e['market'].std()

#%% 5

MU = [i for i in np.arange(0,4.5,.01)]
SIGMA = [f(m) for m in MU]

plt.plot(SIGMA, MU);
plt.scatter(X.std(), X.mean());
plt.scatter([np.sqrt(A/B**2), np.sqrt(1/C), 0, vol_tangency, vol_market], [A/B, B/C, rf, mu_tangency, mu_market], color='k');
plt.text(vol_market, mu_market, 'market portfolio', verticalalignment='top', horizontalalignment='left')
plt.axis([0, 9, 0, 5]);
plt.plot([0, 9], [rf, rf+(mu_tangency-rf)/vol_tangency*9],);

#%% 6
ols = sm.OLS(X_e, sm.add_constant(Y['Mkt-RF'])).fit()
out = ols.params.T
out['std error const'] = np.sqrt(np.diag(np.dot(ols.resid.T, ols.resid)/ols.df_resid))/np.sqrt(T)
out['std error Mkt-RF'] = out['std error const']*np.sqrt(T)/np.sqrt(np.dot(Y['Mkt-RF'].T, Y['Mkt-RF']))

pyperclip.copy(out.round(3).to_latex())

#%% 7
alpha = pd.Series(name = 'alpha', index = X.columns)
resid = pd.DataFrame(index = X.index, columns = X.columns)

for portfolio in X:
    ols = sm.OLS(Y['Mkt-RF'], sm.add_constant(X[portfolio]-Y.RF)).fit()
    resid[portfolio] = ols.resid
    alpha[portfolio] = ols.params['const']

SigmaInv = np.linalg.inv(np.dot(resid.T,resid)/(T-2))
q11 = (Y['Mkt-RF']**2).mean()/T/((Y['Mkt-RF']**2).mean()-Y['Mkt-RF'].mean())
# F statistic
z = (T-n-1)/(T-2)/n/q11*np.dot(np.dot(alpha.T, SigmaInv), alpha.T)
# p-value of statistic
1-stats.f.cdf(z, n, T-n-1)
# confidence region right bound
stats.f.ppf(.95, n, T-n-1)
