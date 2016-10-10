import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, style
style.use('ggplot')
from statsmodels import api as sm
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
from scipy import stats
import pyperclip

# # Part A.1

plt.close('all')

X = pd.read_csv('Average Value Weighted Returns -- Monthly.csv', index_col = 0);
Y = pd.read_csv('market_data.csv', index_col = 0)

X.index = pd.to_datetime(X.index, format='%Y%m')
Y.index = pd.to_datetime(Y.index, format='%Y%m')
T,n = X.shape
#market portfolio
market_returns = Y['Mkt-RF']+Y['RF']
mu_market = market_returns.mean()
vol_market = market_returns.std()

X['market'] = market_returns
X_e = X.subtract( Y['RF'], axis = 0)
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
MU_inefficient = [m for m in MU  if m < B/C]
MU_efficient = [m for m in MU  if m > B/C]
SIGMA_inefficient = [f(m) for m in MU_inefficient]
SIGMA_efficient = [f(m) for m in MU_efficient]

plt.figure(figsize=(10,4))
plt.plot(SIGMA_inefficient, MU_inefficient,c='r');
plt.plot(SIGMA_efficient, MU_efficient,c='b');
plt.plot([0, 9], [rf, rf+(mu_tangency-rf)/vol_tangency*9], c='b')
plt.scatter(X.std(), X.mean(), c='g');
plt.scatter([np.sqrt(A/B**2), np.sqrt(1/C), 0, vol_tangency, vol_market], [A/B, B/C, rf, mu_tangency, mu_market], color='k');
plt.text(vol_market, mu_market, 'market portfolio', verticalalignment='top', horizontalalignment='left')
plt.axis([0, 9, 0, 5]);
plt.xlabel('volatility')
plt.ylabel('return')
plt.tight_layout()

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

##%% # # Part A.2 # #
#
#
##%% 2
#factors = pd.DataFrame(index=X.index,columns=['SMB','HML'])
#factors.SMB = X[['SMALL LoBM', 'ME1 BM2', 'ME1 BM3', 'ME1 BM4', 'SMALL HiBM']].mean(axis=1) - \
#        X[['BIG LoBM', 'ME5 BM2', 'ME5 BM3', 'ME5 BM4', 'BIG HiBM']].mean(axis=1)
#factors.HML = X[['SMALL HiBM', 'ME2 BM5', 'ME3 BM5', 'ME4 BM5', 'BIG HiBM']].mean(axis=1) - \
#        X[['SMALL LoBM', 'ME2 BM1', 'ME3 BM1', 'ME4 BM1', 'BIG LoBM']].mean(axis=1)
#factors['const'] = 1
#factors['market'] = X_e.market
#        
## factors.index = pd.to_datetime(factors.index, format='%Y%m')
#factors.describe()        
#fig = plt.figure();
#ax=fig.add_subplot(321);factors.iloc[:,1].plot(ax=ax);ax.set_title(factors.columns[1]);
#ax=fig.add_subplot(323);factors.iloc[:,0].plot(ax=ax);ax.set_title(factors.columns[0]);
#ax=fig.add_subplot(325);factors.iloc[:,3].plot(ax=ax);ax.set_title(factors.columns[3]);
#    
#ax=fig.add_subplot(322);factors.iloc[:,1].hist(bins=20,ax=ax);ax.set_title(factors.columns[1]);
#ax=fig.add_subplot(324);factors.iloc[:,0].hist(bins=20,ax=ax);ax.set_title(factors.columns[0]);
#ax=fig.add_subplot(326);factors.iloc[:,3].hist(bins=20,ax=ax);ax.set_title(factors.columns[3]);
##factors.hist(bins=20);
#stats.jarque_bera(factors.SMB)
#stats.chi2.ppf(.999,2)
#
##%% 3
#'''Regress the hedge portfolio returns on the excess market returns and a constant, and
#discuss the outcomes. What do they mean? '''
#sm.OLS(factors.SMB, sm.add_constant(X_e.market)).fit(cov_type='HC1').summary()
#sm.OLS(factors.HML, sm.add_constant(X_e.market)).fit(cov_type='HC1').summary()
#stats.t.ppf(.9995,623)
#stats.t.cdf(-8.08,623)*2
##%% 4
#''' Regress the portfolio returns on the excess market returns and the hedge portfolio(s).
#Discuss the outcomes.'''
## 
#ols = sm.OLS(X_e.drop('market', axis=1), factors).fit()
#ols.params
#sols = pd.DataFrame(index = pd.MultiIndex.from_product([factors.columns, ['param', 'std', 'rsq']]), columns = X.drop('market', axis=1).columns, dtype=np.float64)
#sols = sols.sort_index(level=[0,1])
#rsqs = pd.Series(index = X_e.columns)
#for column in sols.columns:
#    temp = sm.OLS(X_e[column], factors).fit()
#    sols.loc[(slice(None), 'param'),column] = pd.DataFrame(temp.params).set_index(pd.MultiIndex.from_product([temp.params.index, ['param']]))[0]
#    sols.loc[(slice(None), 'std'),column] = pd.DataFrame(temp.HC0_se).set_index(pd.MultiIndex.from_product([temp.HC0_se.index, ['std']]))[0]
#    rsqs[column] = temp.rsquared
#
#sols.xs((slice(None),'param')).T.round(3)
#sols.xs((slice(None),'std')).T.round(3)
#sols.round(3).T
#sols.xs((slice(None),'param')).divide( \
#sols.xs((slice(None),'std')),  ).round(3).abs().le(stats.t.ppf(.995,623)).T
#
#np.reshape(sols.xs(('market','param')).round(3),(5,5),'F')
##%% 5
#''' Conduct the GRS-test. What does the outcome mean? Do the hedge portfolio(s)
#capture the effect of the portfolio sort?'''
#ols = sm.OLS(X_e.drop(['SMALL LoBM','BIG LoBM'], axis=1), factors).fit()
#k = ols.df_model
#f_bar = factors.drop('const', axis=1).mean()
#OhmInv = np.linalg.inv(factors.drop('const', axis=1).cov())
#alph = ols.params.T.const
#SigmaInv = np.linalg.inv(np.matmul(ols.resid.T,ols.resid))*(ols.df_model+ols.df_resid)/ols.df_resid
#z = (T-n-k)/n/(1+np.matmul(np.matmul(f_bar.T, OhmInv), f_bar))*np.matmul(np.matmul(alph.T, SigmaInv), alph) 
#print('''
#statistic value:  {:.4f}
#p-value of stat:  {:.4f}
#confidence bound: {:.4f}'''.format(z, 1-stats.f.cdf(z, n, T-n-1), stats.f.ppf(.95, n, T-n-1)))

#%% # # Part A.4 # #


#%% 1

cons = pd.read_excel('consumption_data.xls', index_col = 0)
plt.figure(6)
plt.hist(cons['PCE growth'])
plt.figure(7)
plt.plot(cons['PCE growth'])

cons['PCE growth'].describe()
JB_cons = sms.jarque_bera(cons['PCE growth'])


#%% 2

cons_ols = sm.OLS(X_e.drop('market', axis=1), sm.add_constant(cons['PCE growth'])).fit()
params = cons_ols.params.loc['PCE growth']



#%% 3

mu_e_df = X_e.mean()
mu_e_df = mu_e_df.iloc[0:25]
params.index = mu_e_df.index
cons_ols2 = sm.OLS(mu_e_df, params).fit()
cons_ols2.summary()
#Need corrected t values.


#%% 4

cons_ols3 = sm.OLS(X_e['market'], cons['PCE growth']).fit()
cons_ols3.summary()