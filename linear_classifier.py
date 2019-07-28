#%%
import pandas as pd


filename = "data/polish_sentiment_Dataset.csv"

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

# get only good comments
rate_good = dataset[dataset['rate']==1]

# sample from good comments, 
rate_good.sample(n=10)

#%%