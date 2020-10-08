import pandas as pd
import datetime
import re
import emoji


#loading the dataframes
spray = pd.read_csv(r'C:\Users\kabii\Desktop\FOLDERS\VIT\4th Year\SEM 1\ECE4032 - Neural Network and Deep Learning\EPJ\NNDL_epj\1 - Web Scraping\Amazon Web Scraper\filename_spray.csv')
squeeze = pd.read_csv(r'C:\Users\kabii\Desktop\FOLDERS\VIT\4th Year\SEM 1\ECE4032 - Neural Network and Deep Learning\EPJ\NNDL_epj\1 - Web Scraping\Amazon Web Scraper\filename_squeeze.csv')
press = pd.read_csv(r'C:\Users\kabii\Desktop\FOLDERS\VIT\4th Year\SEM 1\ECE4032 - Neural Network and Deep Learning\EPJ\NNDL_epj\1 - Web Scraping\Amazon Web Scraper\filename_press.csv')
spray2 = pd.read_csv(r'C:\Users\kabii\Desktop\FOLDERS\VIT\4th Year\SEM 1\ECE4032 - Neural Network and Deep Learning\EPJ\NNDL_epj\1 - Web Scraping\Amazon Web Scraper\filename_spray2.csv')

#data cleaning and formatting
spray2.dropna()
squeeze.dropna()
spray.dropna()
press.dropna()

#declarations
i = 0
n1 = len(spray['Review_text'])
n2 = len(squeeze['Review_text'])
n3 = len(press['Review_text'])
n4 = len(spray2['Review_text'])


#coverting date collumn to datetime
for i in range(0,len(spray['Date'])):
    spray['Date'][i] = spray['Date'][i].replace("Reviewed in India on ","")

for l in range(0,len(spray2['Date'])):
    spray2['Date'][l] = spray2['Date'][l].replace("Reviewed in India on ","")

for j in range(0,len(squeeze['Date'])):
    squeeze['Date'][j] = squeeze['Date'][j].replace("Reviewed in India on ","")

for k in range(0,len(press['Date'])):
    press['Date'][k] = press['Date'][k].replace("Reviewed in India on ","")


spray['Date'] = pd.to_datetime(spray['Date'], format='%d %B %Y')
spray2['Date'] = pd.to_datetime(spray2['Date'], format='%d %B %Y')
squeeze['Date'] = pd.to_datetime(squeeze['Date'], format='%d %B %Y')
press['Date'] = pd.to_datetime(press['Date'], format='%d %B %Y')


#removing special charcters like 'Â' etc.
for i in range(n1):
    re.sub("&amp;", "and", spray['Review_text'][i])
    re.sub("&amp;", "and", spray['Review_title'][i])
    re.sub("â€™", "'", spray['Review_text'][i])
    re.sub("â€™", "'", spray['Review_title'][i])
    re.sub("Â|â|€|™|ðŸ|»|ðŸ˜|$|â€|ð|Ÿ|¤|‘|»|âœ|ðŸ™‚|.ðŸ‘", "", spray['Review_text'][i])
    re.sub("Â|â|€|™|ðŸ|»|ðŸ˜|$|â€|ð|Ÿ|¤|‘|»|âœ|ðŸ™‚|.ðŸ‘", "", spray['Review_title'][i])


for i in range(n4):
    re.sub("&amp;", "and", spray2['Review_text'][i])
    re.sub("&amp;", "and", spray2['Review_title'][i])
    re.sub("â€™", "'", spray2['Review_text'][i])
    re.sub("â€™", "'", spray2['Review_title'][i])
    re.sub("Â|â|€|™|ðŸ|»|ðŸ˜|$|â€|ð|Ÿ|¤|‘|»|âœ|ðŸ™‚|.ðŸ‘", "", spray2['Review_text'][i])
    re.sub("Â|â|€|™|ðŸ|»|ðŸ˜|$|â€|ð|Ÿ|¤|‘|»|âœ|ðŸ™‚|.ðŸ‘", "", spray2['Review_title'][i])


    
for i in range(n2):
    re.sub("&amp;", "and", squeeze['Review_text'][i])
    re.sub("&amp;", "and", squeeze['Review_title'][i])
    re.sub("â€™", "'", squeeze['Review_text'][i])
    re.sub("â€™", "'", squeeze['Review_title'][i])
    re.sub("Â|â|€|™|ðŸ|»|ðŸ˜|$|â€|ð|Ÿ|¤|‘|»|âœ|ðŸ™‚|.ðŸ‘", "", squeeze['Review_text'][i])
    re.sub("Â|â|€|™|ðŸ|»|ðŸ˜|$|â€|ð|Ÿ|¤|‘|»|âœ|ðŸ™‚|.ðŸ‘", "", squeeze['Review_title'][i])


    
for i in range(n3):
    re.sub("&amp;", "and", press['Review_text'][i])
    re.sub("&amp;", "and", press['Review_title'][i])
    re.sub("â€™", "'", press['Review_text'][i])
    re.sub("â€™", "'", press['Review_title'][i])
    re.sub("Â|â|€|™|ðŸ|»|ðŸ˜|$|â€|ð|Ÿ|¤|‘|»|âœ|ðŸ™‚|.ðŸ‘", "", press['Review_text'][i])
    re.sub("Â|â|€|™|ðŸ|»|ðŸ˜|$|â€|ð|Ÿ|¤|‘|»|âœ|ðŸ™‚|.ðŸ‘", "", press['Review_title'][i])


#Removing Tabs, newline & extra Spaces
spray.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["",""], regex=True, inplace=True)
spray2.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["",""], regex=True, inplace=True)
squeeze.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["",""], regex=True, inplace=True)
press.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["",""], regex=True, inplace=True)

spray['Review_text'] = spray['Review_text'].str.split().str.join(" ")
spray['Review_title'] = spray['Review_title'].str.split().str.join(" ")

spray2['Review_text'] = spray2['Review_text'].str.split().str.join(" ")
spray2['Review_title'] = spray2['Review_title'].str.split().str.join(" ")

squeeze['Review_text'] = squeeze['Review_text'].str.split().str.join(" ")
squeeze['Review_title'] = squeeze['Review_title'].str.split().str.join(" ")

press['Review_text'] = press['Review_text'].str.split().str.join(" ")
press['Review_title'] = press['Review_title'].str.split().str.join(" ")

#Removing emojis
def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F700-\U0001F77F"  # alchemical symbols
        u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA00-\U0001FA6F"  # Chess Symbols
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        u"\U00002702-\U000027B0"  # Dingbats
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

spray['Review_text'].apply(deEmojify)
spray['Review_title'].apply(deEmojify)
spray2['Review_text'].apply(deEmojify)
spray2['Review_title'].apply(deEmojify)
squeeze['Review_text'].apply(deEmojify)
squeeze['Review_title'].apply(deEmojify)
press['Review_text'].apply(deEmojify)
press['Review_title'].apply(deEmojify)

for i in range(n1):
    spray['Review_text'][i] = emoji.demojize(spray['Review_text'][i]).replace(":"," ")
    spray['Review_title'][i] = emoji.demojize(spray['Review_title'][i]).replace(":"," ")

for i in range(n4):
    spray2['Review_text'][i] = emoji.demojize(spray2['Review_text'][i]).replace(":"," ")
    spray2['Review_title'][i] = emoji.demojize(spray2['Review_title'][i]).replace(":"," ")

for i in range(n2):
    squeeze['Review_text'][i] = emoji.demojize(squeeze['Review_text'][i]).replace(":"," ")
    squeeze['Review_title'][i] = emoji.demojize(squeeze['Review_title'][i]).replace(":"," ")

for i in range(n3):
    press['Review_text'][i] = emoji.demojize(press['Review_text'][i]).replace(":"," ")
    press['Review_title'][i] = emoji.demojize(press['Review_title'][i]).replace(":"," ")

print(spray.head())
print(spray2.head())
print(squeeze.head())
print(press.head())

print(spray.dtypes)
print(spray2.dtypes)
print(squeeze.dtypes)
print(press.dtypes)


#export data to a new file
spray_sanitized = spray.to_csv(r'C:\Users\kabii\Desktop\FOLDERS\VIT\4th Year\SEM 1\ECE4032 - Neural Network and Deep Learning\EPJ\NNDL_epj\sanitized\spray_sanitized.csv')
spray2_sanitized = spray2.to_csv(r'C:\Users\kabii\Desktop\FOLDERS\VIT\4th Year\SEM 1\ECE4032 - Neural Network and Deep Learning\EPJ\NNDL_epj\sanitized\spray2_sanitized.csv')
squeeze_sanitized = squeeze.to_csv(r'C:\Users\kabii\Desktop\FOLDERS\VIT\4th Year\SEM 1\ECE4032 - Neural Network and Deep Learning\EPJ\NNDL_epj\sanitized\squeeze_sanitized.csv')
press_sanitized = press.to_csv(r'C:\Users\kabii\Desktop\FOLDERS\VIT\4th Year\SEM 1\ECE4032 - Neural Network and Deep Learning\EPJ\NNDL_epj\sanitized\press_sanitized.csv')