
# coding: utf-8



import pandas as pd
import numpy as np

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:85% !important; }</style>"))

pd.set_option("display.max_columns",50)




regular_df = pd.read_csv('data/DataFiles/RegularSeasonDetailedResults.csv')
regular_df.columns = [i.lower() for i in regular_df.columns]
team_df =  pd.read_csv('data/DataFiles/Teams.csv')
team_df.columns = [i.lower() for i in team_df.columns]
teamcoach_df = pd.read_csv('data/DataFiles/TeamCoaches.csv')
teamcoach_df.columns = [i.lower() for i in teamcoach_df.columns]
post_df = pd.read_csv('data/DataFiles/NCAATourneyCompactResults.csv')
post_df.columns = [i.lower() for i in post_df.columns]




teamcoach_df.head()




combine = teamcoach_df.pipe(lambda x:x.assign(daysexp = x.lastdaynum-x.firstdaynum))




combine.head()









post_df.head()




post_df.daynum.max()




champions_df = post_df.query('daynum==154')[['season','daynum','wteamid']]




champions_df.columns = [['season','lastdaynum','teamid']]
champions_df['champions'] = [1 for i in champions_df.teamid]




champions_df.head()




combine2 = pd.merge(combine, champions_df, how='left',on=['season','lastdaynum','teamid'])




combine2.champions = combine2.champions.fillna(0)




combine2.head()




intermediate = pd.DataFrame({'season':post_df.groupby(['season'])['daynum'].min().index.values, 'playoff_startday':[v for i,v in enumerate(post_df.groupby(['season'])['daynum'].min())]})




postmod_df = pd.merge(post_df,intermediate,how='left',on='season')




postmod_df.head()




postmod_df.shape




intermediate = postmod_df.groupby(['season','wteamid'])['daynum'].count().reset_index()
intermediate['count'] = [1 for i in intermediate.daynum]




intermediate2 = postmod_df.groupby(['season','lteamid'])['daynum'].count().reset_index()
intermediate2['count'] = [1 for i in intermediate2.daynum]
intermediate2.columns = [['season','wteamid','daynum','count']]




intermediate3 = pd.concat([intermediate, intermediate2], axis =0)




intermediate3.head()




intermediate3.drop_duplicates(subset=['wteamid','season'], inplace = True)




intermediate3.drop('daynum', axis=1, inplace=True)




intermediate3.columns = [['season','teamid','playoff']]




intermediate3.head()




combine2.head()




combine3 = pd.merge(combine2, intermediate3, how='left', on=['season','teamid'])




combine3.playoff = combine3.playoff.fillna(0)




regular_df.head()




teamcoach_df2.query('lastdaynum<154 & season==2005')




teamcoach_df2 = teamcoach_df
teamcoach_df2.columns = [['season','wteamid','firstdaynum','lastdaynum','wcoachname']]




regularmod_df = pd.merge(regular_df,teamcoach_df2,how='left',on=['season','wteamid'])
regularmod_df.head()




regular_df.season.unique().shape




reg_int = regular_df.groupby(['season','wteamid'])['index'].count().reset_index()
reg_int.columns = [['season','teamid','wcount']]
reg_int.head()




reg_int2 = regular_df.groupby(['season','lteamid'])['daynum'].count().reset_index()
reg_int2.columns = [['season','teamid','lcount']]
reg_int2.head()




reg_winrate =(
        pd.merge(reg_int2,reg_int,how='outer',on=['season','teamid']).fillna(0)
        .pipe(lambda x: x.assign(winrate=x.wcount/(x.wcount+x.lcount)))
)
reg_winrate.columns = [['season','teamid','lcount','wcount','reg_winrate']]




reg_winrate.head()




post_int = (
            post_df
            .groupby(['season','wteamid'])['daynum']
            .count()
            .reset_index() 
)
post_int.columns = [['season','teamid','wcount']]




post_int.head()




post_int2 = (
            post_df
            .groupby(['season','lteamid'])['daynum']
            .count()
            .reset_index() 
)
post_int2.columns = [['season','teamid','lcount']]




post_winrate= (
                post_int
                .merge(post_int2,how='outer',on=['season','teamid'])
                .fillna(0)
                .pipe(lambda x: x.assign(winrate=x.wcount/(x.wcount+x.lcount)))
)
post_winrate.columns = [['season','teamid','wcount','lcount','post_winrate']]




post_winrate.head()




combine3.head()




combine4 = (
            combine3
            .merge(reg_winrate,how='left',on=['season','teamid'])
            .drop(['lcount','wcount'],axis = 1)
            .merge(post_winrate,how='left',on=['season','teamid'])
            .drop(['lcount','wcount'],axis=1)
            .fillna(0)
)




combine4.head()

