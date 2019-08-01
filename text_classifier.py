#%%
import pandas as pd
import numpy as np

from sklearn import svm, metrics

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


filename = './data/polish_sentiment_shop_comments.csv'
df = pd.read_csv(filename)

#%%
print('spliting dataset')
x_train, x_test, y_train, y_test = train_test_split(
    df['description'], df['rate'], random_state=0, test_size=0.2)


# find TfIdf model on train and transform test set
tfidf = TfidfVectorizer(max_features=20000, max_df=0.9, min_df=20)
x_train_tfidf = tfidf.fit_transform(x_train)
x_test_tfidf = tfidf.transform(x_test)
#%%


clf = svm.SVC(C=1, gamma=0.01)
#clf = SGDClassifier()
#clf = RandomForestClassifier(n_estimators=30)


import datetime as dt
# We learn the digits on train part
start_time = dt.datetime.now()
print(f'Start learning at {start_time}')

clf.fit(x_train_tfidf, y_train)

end_time = dt.datetime.now() 
print(f'Stop learning {end_time}')
elapsed_time= end_time - start_time
print(f'Elapsed learning {elapsed_time}')


#%%
expected = y_test
predicted = clf.predict(x_test_tfidf)

print(f'Classification report for classifier: \n{clf}')
print(f'Results\n{metrics.classification_report(expected, predicted)}')      

#%%
