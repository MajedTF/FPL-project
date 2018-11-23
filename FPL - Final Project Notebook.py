
# coding: utf-8

# In[97]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


gw0 = pd.read_csv("FPL_2018_19_Wk0.csv")  ##Previous season data
gw1 = pd.read_csv("FPL_2018_19_Wk1.csv")
gw2 = pd.read_csv("FPL_2018_19_Wk2.csv")
gw3 = pd.read_csv("FPL_2018_19_Wk3.csv")
gw4 = pd.read_csv("FPL_2018_19_Wk4.csv")
gw5 = pd.read_csv("FPL_2018_19_Wk5.csv")
gw6 = pd.read_csv("FPL_2018_19_Wk6.csv")
gw7 = pd.read_csv("FPL_2018_19_Wk7.csv") ##Each GW is updated as the information is cumulative through gameweeks


# # Columns info and Terminlogoy
# 1. Name: Player's name
# 2. Team: Team's name
# 3. Position: Player's position
# 4. Cost: Player's value in FPL
# 5. Influence is the first measurement - this evaluates the degree to which that player has made an impact on a match.
# 6. Creativity assesses player performance in terms of producing goal scoring opportunities for others. 
# 7. Threat is a measurement where producing a value that examines a player’s threat on goal; it therefore gauges those individuals most likely to score goals.
# 8. The ICT Index is a football statistical index developed specifically to assess a player as an FPL asset. It uses match event data to generate a single score for three key areas – Influence, Creativity and Threat.
# 9. Goals Conceded - how many goals a team/player conceded throughout the season
# 10. Goals Scored - how many goals a player scored throughout the season
# 11. Own Goals - how many own goals a player scored against his own team throughout the season
# 12. Penalties missed - the number of penalties missed
# 13. Penalties scored - the number of penalties scored
# 14. Penalties saved - for Goal Keepers how many penalties has he saved.
# 15. Saves - number of saved made by a GK
# 16. Yellow Cards - number of yellow cards a player had
# 17. Red Cards - number of red cards a player had
# 18. TSB - Team Selected By% , a percentage of ownership by FPL users.
# 19. Minutes - minutes played throughout the season
# 20. Bonus - Bonus points gained each match for players who perform well.
# 21. Points - Number of points gained throughout the season. --> Model's target (y)
# 
# 

# In[136]:


pd.options.display.max_columns=100


# In[137]:


gw7.head()


# In[25]:


gw7.head()
gw7.tail()
gw7.dtypes


# In[145]:


list(gw7.select_dtypes(include=object))


# In[146]:


gw7.shape


# In[31]:


gw7.Position.value_counts(dropna=False)


# In[74]:


gw7.Team.value_counts()
#Not very informative but gives me insights on how many players are registered


# In[148]:


gw7.sort_values(by='Threat',ascending=False)[:10]


# In[147]:


gw7.sort_values(by='Creativity',ascending=False)[:10]
#Last two lines showed who are the players that contribute the most in -
#chance creation (Creativity) and goal scoring oppurtunities (Threat)


# In[149]:


gw7.sort_values(by='Threat',ascending=False)[['Name', 'Team', 'Position', 'Creativity','Threat','Points']][:10]


# In[167]:


Top_gw7 = gw7.sort_values('Points', ascending=False)

Top_gw7.groupby('Position').head(3)[['Position', 'Name', 'Team', 'Points']]

#Top 3 Players in each position


# In[140]:


gw7.isnull().sum()[gw7.isnull().sum() != 0]
#No Missing Values


# In[78]:


gw7[gw7['Points']==0].shape
#158 players have not been playing, or even injured 


# In[141]:


print(gw7[gw7['Points'] < 0].shape)

gw7[gw7['Points'] < 0][['Name', 'Points']]


# In[169]:


gw7[gw7['Points'] < 0][['Name', 'Points', 'Goals_conceded', 'Minutes']]


# In[142]:


for col in gw7: print(col, gw7[col].nunique())


# In[ ]:


for col in gw7._get_numeric_data(): print(col, "{:4f}".format(gw7[col].skew()))


# In[72]:


constant_players = gw7[gw7['Minutes'] >= 270]
print(constant_players)
#Players who played more than 3 matches


# In[135]:


constant_players.shape


# In[144]:


constant_players.describe().T.style


# In[143]:


gw7.corr().style


# In[131]:


correlations = gw7.corr()
correlations.style
plt.figure(figsize=(18, 18))
sns.heatmap(correlations, annot=True, linewidth=0.4)
plt.yticks(rotation=0);


# In[73]:


sns.lmplot(x='Threat', y='Points', data=constant_players)
plt.show()
sns.lmplot(x='Influence', y='Points', data=constant_players)
plt.show()
sns.lmplot(x='Creativity', y='Points', data=constant_players)
plt.show()


# In[150]:


gw7.columns


# In[134]:


sns.distplot(constant_players['Points']);


# In[138]:


sns.distplot(gw7['Points']);
#Positively skewed

