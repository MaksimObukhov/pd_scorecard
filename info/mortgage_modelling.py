#%% 0. Settings
# imports
import pandas as pd
import numpy as np
import statsmodels.api as sm
from optbinning import OptimalBinning, BinningProcess
from sklearn import metrics
import matplotlib.pyplot as plt


#%% 1. Data Import
data = pd.read_csv("mortgage_sample.csv").dropna()

#%% 2. Sample preparation and checks
# Don't forget to construct the target and sample appropriately!
sample = data[(data["sample"] == "public") & ((data["time"] - data["first_time"]) % 12 == 0)]
sample.head()
#%%
drs = sample.groupby("time")["default_time"].agg(["count", "mean"]).reset_index()

fig, ax = plt.subplots(figsize=(20,10)) 

drs.plot(x = 'time', y = 'count', ax = ax) 
drs.plot.bar(x = 'time', y = 'mean', ax = ax, secondary_y = True) 

#%% 3. Predictor preparation
#%%
predictor = "LTV_orig_time"
x = sample[predictor]
y = sample["default_time"]

optb = OptimalBinning(name=predictor)
optb.fit(x, y)
optb.binning_table.build()
#%%
optb.binning_table.plot(metric="event_rate")
optb.binning_table.analysis()

#%%
predictor = "LTV_orig_time"
x = sample[predictor]
y = sample["TARGET"]

optb = OptimalBinning(name=predictor, monotonic_trend = "auto_asc_desc")
optb.fit(x, y)
optb.binning_table.build()
#%%
optb.binning_table.plot(metric="event_rate")
optb.binning_table.analysis()

#%%
predictor = "LTV_orig_time"
x = np.clip(sample[predictor], a_min = None, a_max = 92.75)
y = sample["TARGET"]

optb = OptimalBinning(name=predictor, special_codes = [92.75])
optb.fit(x, y)
optb.binning_table.build()
#%%
optb.binning_table.plot(metric="event_rate")
optb.binning_table.analysis()


#%% Binning Process
predictors = ["FICO_orig_time", "LTV_time"]
binning_process = BinningProcess(variable_names = predictors)

X = sample[predictors]
y = sample["default_time"]

binning_process.fit_transform(X, y)
binning_process.summary()

#%% 4. Regression estimation
X = sm.add_constant(X.dropna())

#%%
logit_mod = sm.Logit(endog=y, exog=X)
estimated_model = logit_mod.fit(disp=0)
estimated_model.summary()


#%%


#%% start with intercept
selected = ['const']
not_selected = [key for key in predictors if key not in selected]
p_value = 0.01

#%% Implement forward regression
while True:
    # model changed?
    changed = False
    selected_prev = selected.copy()  # save previoust list

    # dataframe with p-values
    p_values = pd.Series(index=not_selected, dtype='float64')

    ## forward step
    # loop over not selected columns
    for key in not_selected:
        # estimate model
        logit_mod = sm.Logit(endog=y, exog=X[selected + [key]])
        estimated_model = logit_mod.fit(disp=0)
                        
        # extract p_value
        p_values[key] = estimated_model.pvalues[key]

    # select best predictor with best p-value (if it is below threshold)
    min_pval = p_values.min()
    if min_pval < p_value:
        best_feature = p_values.idxmin()
        selected.append(best_feature)
        not_selected.remove(best_feature)
        print(
            'Add  {:30} with p-value {:.6}'.format(best_feature, min_pval))
    else:
        print('No predictor added')

    # check if model changed
    if set(selected) != set(selected_prev):
        changed = True

    if len(selected) == len(X.columns):
        print("\n No more predictors to test!")
        break

    if not changed:
        break

#%% 5. Estimate final model
logit_mod = sm.Logit(endog=y, exog=X[selected])
estimated_model = logit_mod.fit(disp=0)
y_pred = estimated_model.predict()
estimated_model.summary()
#%% 6. Performance assessment
fpr, tpr, _ = metrics.roc_curve(y,  y_pred)
auc = metrics.roc_auc_score(y, y_pred)
plt.plot(fpr,tpr,label="GINI="+str(2*auc-1))
plt.legend(loc=4)
plt.show()

