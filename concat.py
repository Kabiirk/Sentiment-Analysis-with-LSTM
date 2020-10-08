import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 

senti_obj = SentimentIntensityAnalyzer()

squeeze = pd.read_csv(r'C:\Users\kabii\Desktop\FOLDERS\VIT\4th Year\SEM 1\ECE4032 - Neural Network and Deep Learning\EPJ\NNDL_epj\sanitized_sentiment\squeeze_sanitized_sentiment.csv')
press = pd.read_csv(r'C:\Users\kabii\Desktop\FOLDERS\VIT\4th Year\SEM 1\ECE4032 - Neural Network and Deep Learning\EPJ\NNDL_epj\sanitized_sentiment\press_sanitized_sentiment.csv')
spray2 = pd.read_csv(r'C:\Users\kabii\Desktop\FOLDERS\VIT\4th Year\SEM 1\ECE4032 - Neural Network and Deep Learning\EPJ\NNDL_epj\sanitized_sentiment\spray2_sanitized_sentiment.csv')

print(squeeze)
print("\n")
print(press)
print("\n")
print(spray2)

stack = pd.concat([squeeze, press, spray2], ignore_index=True)
stack.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis='columns', inplace=True)
print(stack)

st = stack.to_csv(r'C:\Users\kabii\Desktop\FOLDERS\VIT\4th Year\SEM 1\ECE4032 - Neural Network and Deep Learning\EPJ\NNDL_epj\concat.csv')
