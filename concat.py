import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 

senti_obj = SentimentIntensityAnalyzer()

squeeze = pd.read_csv(r'C:\Users\username\..\sanitized_sentiment\squeeze_sanitized_sentiment.csv')
press = pd.read_csv(r'C:\Users\username\..\sanitized_sentiment\press_sanitized_sentiment.csv')
spray2 = pd.read_csv(r'C:\Users\username\..\sanitized_sentiment\spray2_sanitized_sentiment.csv')

print(squeeze)
print("\n")
print(press)
print("\n")
print(spray2)

stack = pd.concat([squeeze, press, spray2], ignore_index=True)
stack.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis='columns', inplace=True)
print(stack)

st = stack.to_csv(r'C:\Users\username\..\concat.csv')

#Functionality to upload files directly to drive, modify filepath as required
# pip install pydrive
from pydrive.drive import GoogleDrive 
from pydrive.auth import GoogleAuth 
   
# For using listdir() 
import os 
   
  
# Below code does the authentication 
# part of the code 
gauth = GoogleAuth() 
  
# Creates local webserver and auto 
# handles authentication. 
gauth.LocalWebserverAuth()        
drive = GoogleDrive(gauth) 
   
# replace the value of this variable 
# with the absolute path of the directory 
path = r"C:\Users\username\..\concat.csv"   
   
# iterating thought all the files/folder 
# of the desired directory 
for x in os.listdir(path): 
   
    f = drive.CreateFile({'title': x}) 
    f.SetContentFile(os.path.join(path, x)) 
    f.Upload() 
  
    # Due to a known bug in pydrive if we  
    # don't empty the variable used to 
    # upload the files to Google Drive the 
    # file stays open in memory and causes a 
    # memory leak, therefore preventing its  
    # deletion 
    f = None

    
    
    
    
    
    
    
    
#credits : https://www.geeksforgeeks.org/uploading-files-on-google-drive-using-python/
