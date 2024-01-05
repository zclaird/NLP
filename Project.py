import pandas as pd
import numpy as np
import seaborn as sns
import random
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import LatentDirichletAllocation

# Read data and check text observation
lotr = pd.read_csv('lotr_scripts.csv', names=['Index', 'Character', 'Dialog', 'Movie'])
lotr['Dialog'][59]
lotr['Movie'][100]

# Tidy data
lotr['Dialog'] = lotr['Dialog'].str.replace('\xa0', '').str.strip()
lotr['Dialog'] = lotr['Dialog'].str.replace("'", "")
lotr['Movie'] = lotr['Movie'].str.strip()
lotr = lotr.dropna().reset_index(drop=True)
lotr.head()
lotr['Dialog'][59]
lotr['Movie'][100]

# Playing around with TextBlob
for i in range(1,50):
    print(lotr['Dialog'][i])
line = lotr['Dialog'][48]
textblob_obj = TextBlob(line)
textblob_obj
textblob_obj.words
len(textblob_obj.words)
TextBlob(lotr['Dialog'][48]).sentiment

# Create polarity column
def get_polarity(line):
    return TextBlob(line).sentiment.polarity
get_polarity(lotr['Dialog'][49])
lotr['Polarity'] = lotr["Dialog"].apply(get_polarity)
lotr['Polarity']
len(lotr[lotr["Polarity"] == 0])

# Create -1, 0, or 1 var for classification testing
Polarity_type = []

for i in lotr['Polarity']:
    if i < 0:
        Polarity_type.append(-1)
    elif i == 0:
        Polarity_type.append(0)
    else:
        Polarity_type.append(1)
        
lotr['Polarity_type'] = Polarity_type
lotr.head(50)
lotr.dtypes
lotr['Polarity_type'] = lotr['Polarity_type'].astype(str)
lotr.dtypes

# Split Data
x = lotr['Dialog'].values
y = lotr['Polarity_type'].values
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = .3, 
                                                random_state = 1)

# Bag-of-Words model
vectorizer = CountVectorizer()
vectorizer.fit(xtrain)
vectorizer.vocabulary_
len(vectorizer.vocabulary_)

xfeature_train = vectorizer.transform(xtrain)
xfeature_test = vectorizer.transform(xtest)

# Create model
model = LogisticRegression()
model.fit(xfeature_train, ytrain)
ypred = model.predict(xfeature_test)
accuracy_score(ypred, ytest)
confusion_matrix(ypred, ytest)

misclassified = np.where(ytest != ypred)[0].tolist()
misclassified

correct = np.where(ytest == ypred)[0].tolist()
correct

# Check some on some results and see how we did
def check_class(i):
    print("Actual: {}\nPredicted: {}\nDialog: {} \n".format(ytest[i], ypred[i], xtest[i]))
    
for i in misclassified:
    check_class(i)
    
for i in correct:
    check_class(i)



##### Topic modeling ##########
cv = CountVectorizer(max_df = 0.95, min_df = 5, stop_words = 'english') # see slides for info on max/min df params
dtm = cv.fit_transform(lotr['Dialog']) # document term matrix, see slides for info 
lda = LatentDirichletAllocation(n_components=10, random_state=42)
lda.fit(dtm)

# Get vocab
len(cv.get_feature_names_out()) # we got 484 terms
for i in range(1, 100): # gets like 100 words
    print(cv.get_feature_names_out()[i])

# Get topics
topic_list = []
def topics():
    for i in range(0, len(lda.components_)):
        topic = lda.components_[i]
        top = topic.argsort()[-10:] # sorts the topic probabilities from low to high and then gets the last 10 obs (10 greatest probs)
        print("Topic {}".format(i + 1))
        for j in top:
            topic_list.append(cv.get_feature_names_out()[j])
            print(cv.get_feature_names_out()[j])
        print("\n")
topics()
topic_list # print all topic words

# Create sublists to access each topic, this will help for the output later
sublists = []
sublist_length = 10

# loop to create sublists of size 10
for i in range(0, len(topic_list), sublist_length):
    sublist = topic_list[i:i + sublist_length]
    sublists.append(sublist)
    
topics()
sublists[0]
sublists[4]
    
len(lda.components_)
type(lda.components_)
lda.components_.shape
lda.components_

single_topic = lda.components_[0]
top_ten = single_topic.argsort()[-10:] # get the top 10 words, -[-10:] means go from 10th to last until end

for i in top_ten:
    print(cv.get_feature_names_out()[i]) # test
    
results = lda.transform(dtm)
def get_topic(i):
    print("Topic: {}\n\nWords in Topic: {}\n\nProbability of Topic: {}\n\nArticle: {}".format(np.argmax(results[i]) + 1, 
                                                                                       sublists[np.argmax(results[i])],
                                                                                       np.max(results[i]).round(4), 
                                                                                       lotr["Dialog"][i]))
get_topic(random.randint(0, len(results)))

    
# Check results and try to make sense of things
results = lda.transform(dtm)
results[100]
lotr["Dialog"][100]


###########  NEW DATASET FOR TOPIC MODELING ############################

# All the code here follows the exact same logic as above

# Topic modeling 2
articles = pd.read_csv("Articles.csv", encoding='unicode_escape')
cv = CountVectorizer(max_df = 0.90, min_df = 10, stop_words = 'english')
dtm2 = cv.fit_transform(articles["Article"]) # this turns it into a matrix or something?
dtm2 # shows number of obs in first number of output
lda = LatentDirichletAllocation(n_components=10, random_state=42)
lda.fit(dtm2)

# Get vocab
dtm2 # 2692 articles, 5464 terms
len(cv.get_feature_names_out()) # this val is the same as the second val in line 115
len(lda.components_) # contains 10 probability lists, one for each topic that we specified
lda.components_[0] # probabilities that each of these terms belongs to topic 1

# Run these 2 lines to get a random word from our list
randomID = random.randint(0, len(cv.get_feature_names_out()))
cv.get_feature_names_out()[randomID]

# Get topics
topic_list = []
def topics():
    for i in range(0, len(lda.components_)):
        topic = lda.components_[i]
        top = topic.argsort()[-10:] # sorts the topic probabilities from low to high and then gets the last 10 obs (10 greatest probs)
        print("Topic {}".format(i + 1))
        for j in top:
            topic_list.append(cv.get_feature_names_out()[j])
            print(cv.get_feature_names_out()[j])
        print("\n")
topics()
topic_list # print all topic words

# Create sublists to access each topic
sublists = []
sublist_length = 10

# loop to create sublists of size 10
for i in range(0, len(topic_list), sublist_length):
    sublist = topic_list[i:i + sublist_length]
    sublists.append(sublist)
    
topics()
sublists[0]
sublists[5]
    
# Check results
results = lda.transform(dtm2)
def get_topic(i):
    print("Topic: {}\n\nWords in Topic: {}\n\nProbability of Topic: {}\n\nArticle: {}".format(np.argmax(results[i]) + 1, 
                                                                                       sublists[np.argmax(results[i])],
                                                                                       np.max(results[i]).round(4), 
                                                                                       articles["Article"][i]))

# Run line to cycle through some results
get_topic(random.randint(0, len(results)))