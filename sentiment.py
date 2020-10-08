import pandas as pd
import datetime
import re
import emoji
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 


squeeze = pd.read_csv(r'C:\Users\kabii\Desktop\FOLDERS\VIT\4th Year\SEM 1\ECE4032 - Neural Network and Deep Learning\EPJ\NNDL_epj\sanitized\squeeze_sanitized.csv')
press = pd.read_csv(r'C:\Users\kabii\Desktop\FOLDERS\VIT\4th Year\SEM 1\ECE4032 - Neural Network and Deep Learning\EPJ\NNDL_epj\sanitized\press_sanitized.csv')
spray2 = pd.read_csv(r'C:\Users\kabii\Desktop\FOLDERS\VIT\4th Year\SEM 1\ECE4032 - Neural Network and Deep Learning\EPJ\NNDL_epj\sanitized\spray2_sanitized.csv')

#Corrections in spray2
spray2['Review_text']=spray2['Review_text'].fillna("")
spray2['Review_title']=spray2['Review_title'].fillna("")
'''
print(squeeze.head())
print(press.head())
print(spray2.head())
'''
senti_obj = SentimentIntensityAnalyzer()

Positive = []
Negative = []
Neutral = []
Compound = []
Overall_emotion = []



for i in range(0,len(squeeze['Review_text'])):

    senti_dict = senti_obj.polarity_scores(squeeze['Review_text'][i])

    Positive.append(senti_dict['pos']) 
    Negative.append(senti_dict['neg'])
    Neutral.append(senti_dict['neu'])
    Compound.append(senti_dict['compound'])

    if(senti_dict['compound']>0):
        Overall_emotion.append("Positive")

    elif(senti_dict['compound']<0):
        Overall_emotion.append("Negative")

    else:
        Overall_emotion.append("Neutral")

squeeze['Positive'] = Positive
squeeze['Negative'] = Negative 
squeeze['Neutral'] = Neutral
squeeze['Compound'] = Compound
squeeze['Overall'] = Overall_emotion


Positive2 = []
Negative2 = []
Neutral2 = []
Compound2 = []
Overall_emotion2 = []

for j in range(0,len(press['Review_text'])):

    senti_dict2 = senti_obj.polarity_scores(press['Review_text'][j])

    Positive2.append(senti_dict2['pos']) 
    Negative2.append(senti_dict2['neg'])
    Neutral2.append(senti_dict2['neu'])
    Compound2.append(senti_dict2['compound'])

    if(senti_dict2['compound']>0):
        Overall_emotion2.append("Positive")

    elif(senti_dict2['compound']<0):
        Overall_emotion2.append("Negative")

    else:
        Overall_emotion2.append("Neutral")

press['Positive'] = Positive2
press['Negative'] = Negative2
press['Neutral'] = Neutral2
press['Compound'] = Compound2
press['Overall'] = Overall_emotion2


Positive3 = []
Negative3 = []
Neutral3 = []
Compound3 = []
Overall_emotion3 = []

for k in range(0,len(spray2['Review_text'])):

    senti_dict3 = senti_obj.polarity_scores(spray2['Review_text'][k])

    Positive3.append(senti_dict3['pos']) 
    Negative3.append(senti_dict3['neg'])
    Neutral3.append(senti_dict3['neu'])
    Compound3.append(senti_dict3['compound'])

    if(senti_dict3['compound']>0):
        Overall_emotion3.append("Positive")

    elif(senti_dict3['compound']<0):
        Overall_emotion3.append("Negative")

    else:
        Overall_emotion3.append("Neutral")

spray2['Positive'] = Positive3
spray2['Negative'] = Negative3
spray2['Neutral'] = Neutral3
spray2['Compound'] = Compound3
spray2['Overall'] = Overall_emotion3


'''
print(squeeze.head())
print(press.head())
print(spray2.head())
'''

#sq_wr = squeeze.to_csv(r'C:\Users\kabii\Desktop\FOLDERS\VIT\4th Year\SEM 1\ECE4032 - Neural Network and Deep Learning\EPJ\NNDL_epj\sanitized_sentiment\squeeze_sanitized_sentiment.csv')
#pr_wr = press.to_csv(r'C:\Users\kabii\Desktop\FOLDERS\VIT\4th Year\SEM 1\ECE4032 - Neural Network and Deep Learning\EPJ\NNDL_epj\sanitized_sentiment\press_sanitized_sentiment.csv')
#sp2_wr = spray2.to_csv(r'C:\Users\kabii\Desktop\FOLDERS\VIT\4th Year\SEM 1\ECE4032 - Neural Network and Deep Learning\EPJ\NNDL_epj\sanitized_sentiment\spray2_sanitized_sentiment.csv')
spray = pd.read_csv(r'C:\Users\kabii\Desktop\FOLDERS\VIT\4th Year\SEM 1\ECE4032 - Neural Network and Deep Learning\EPJ\NNDL_epj\sanitized\spray_sanitized.csv')
l = []
for i in range(10):
    s = spray["Review_text"][i]
    s_dict = senti_obj.polarity_scores(s)
    l.append(s_dict['compound'])
    print(s, s_dict['compound'])

