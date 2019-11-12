#%%
import pandas as pd


filename = "./data/polish_sentiment_dataset.csv"

dataset = pd.read_csv(filename, delimiter = ",")
 
# Delete unused column
del dataset['length']

dataset['description'].isnull().sum()
dataset['rate'].isnull().sum()
 
#%% 

# print how many examples
for r in [-1, 0, 1]:
    count_r = (dataset['rate']==r).sum()
    print(f'#[{r}]={count_r}') 

null_desc= dataset['description'].isnull().sum()
print(f'null description {null_desc}')
null_rate = dataset['rate'].isnull().sum()
print(f'null rate {null_rate}')

# Delete All NaN values from columns=['description','rate']
#dataset = dataset[dataset['description'].notnull() & dataset['rate'].notnull()]
 
 
# We set all strings as lower case letters
#dataset['description'] = dataset['description'].str.lower()

#%%
dataset.hist(column='rate', bins=3)

# get only bad comments
rate_bad = dataset[dataset['rate'] == -1]
count_bad = rate_bad.shape


rate_bad['desc_len'] = rate_bad['description'].map(str).apply(len)

rate_bad.hist(column='desc_len', bins=[0,10,20,50,100,200,500,1000])
# count values in bins
rate_bad['desc_len'].value_counts(bins=[0,100,200,30000])

# take descriptions longer then 200 and shorter then 3500, there are aprox. 20K chars 
rb= rate_bad[(rate_bad['desc_len']>=200) & (rate_bad['desc_len']<=3500)]


#%%
# get only good comments
rate_good = dataset[dataset['rate']==1]

rate_good['desc_len'] = rate_good['description'].map(str).apply(len)


rate_good.hist(column='desc_len', bins=[0,10,20,50,100,200,500,10000])
# count values in bins
rate_good['desc_len'].value_counts(bins=[0,100,200, 30000])

# take descriptions logner then 200 chars ~42K
rg= rate_good[(rate_good['desc_len']>=200) & (rate_good['desc_len']<=3500)]
# sample from good comments, 

number_of_bad = rb.shape[0]
rg = rg.sample(number_of_bad)

#%%
# concat two data frames

new_dataset = pd.concat([rb, rg])
new_dataset = new_dataset.reset_index()
del new_dataset['index']
#new_dataset.index.name='id'


# %%

filename = './data/polish_sentiment_shop_comments.csv'
new_dataset.to_csv(filename, index=False)

#%%
# reopen the saved dataset
filename = './data/polish_sentiment_shop_comments.csv'
dataset = pd.read_csv(filename)
#%%
new_dataset.head()

#%%
dataset.head()

#%%
