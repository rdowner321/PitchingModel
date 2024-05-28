from pybaseball import statcast
from pybaseball import statcast_pitcher
from pybaseball import playerid_lookup
from pybaseball.statcast_pitcher import statcast_pitcher_active_spin
from pybaseball.statcast_pitcher import statcast_pitcher_arsenal_stats
from pybaseball.statcast_pitcher import statcast_pitcher_pitch_movement
from pybaseball.statcast_pitcher import statcast_pitcher_spin_dir_comp

import pandas as pd



resultDF=pd.read_csv('pitch-arsenal-stats 2022.csv')

movementDF=pd.read_csv('spin-direction-pitches 2022.csv')

newDF=resultDF.merge(movementDF,on=['player_id','pitch_name','Season'],how='inner')

newDF=newDF.dropna()

newDF.to_csv('fullinput2022.csv')