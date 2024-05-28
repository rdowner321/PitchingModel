from pybaseball import statcast_pitcher_spin
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import MinMaxScaler

import numpy as np
# get data for sinker / slider combos in 2020 using pitch codes and from the batter's POV
#data = statcast_pitcher_spin_dir_comp(2023, pitch_a=4-Seamer, pitch_b=Curveball,minP=20,pitcher_pov=True)

#resultDF= pd.read_csv('resultting.csv')

resultDF=pd.read_csv('Input/fullinput.csv')

print(resultDF)


rr=Ridge(alpha=1)
    
sfs=SequentialFeatureSelector(rr,n_features_to_select=5,direction='forward',n_jobs=4)   

removed_columns=['name','team_name_alt','player_id','pitch_name','pitch_type','run_value_per_100','ID','run_value','pa','pitch_per','pitch_usage','pitches_thrown','pitches','Season']

selected_columns=resultDF.columns[~resultDF.columns.isin(removed_columns)]

scaler=MinMaxScaler()

resultDF.loc[:,selected_columns]=scaler.fit_transform(resultDF[selected_columns])

sfs.fit(resultDF[selected_columns], resultDF['run_value_per_100'])

print(selected_columns[sfs.get_support()])

predictors=list(selected_columns[sfs.get_support()])

def backtest(data, model, predictors,start=1,step=1):
    all_predictions=[]

    years=sorted(data["Season"].unique())

    for i in range(start, len(years), step):

        current_year=years[i]

        train=data[data["Season"]<current_year]
        test=data[data["Season"]==current_year]

        model.fit(train[predictors],train['run_value_per_100'])
    
        preds=model.predict(test[predictors])
        preds=pd.Series(preds, index=test.index)
        combined= pd.concat([test['run_value_per_100'],preds],axis=1)
        combined.columns=["actual","prediction"]
#new commit comment
        all_predictions.append(combined)
    return all_predictions

predictions=backtest(resultDF,rr,predictors)


predDF=pd.DataFrame(predictions[0])

mergedDF=predDF.merge(resultDF,left_index=True,right_index=True)

mergedDF.to_csv('Output/ModelResults.csv')