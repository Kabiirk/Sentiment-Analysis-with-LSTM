import pandas as pd
import seaborn as sns

import tensorflow as tf
import numpy as np
import re
import codecs
import os
from nltk.tokenize import RegexpTokenizer

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Bidirectional
from keras.preprocessing import sequence
from keras.layers import Dropout
import h5py
import utility_functions as uf
from keras.models import model_from_json
from keras.models import load_model
from nltk.tokenize import RegexpTokenizer


#reading the data
spray = pd.read_csv("/sanitized/spray_sanitized.csv")
spray.drop('Unnamed: 0', axis='columns', inplace=True)

def load_embeddings(embedding_path):
  """Loads embedings, returns weight matrix and dict from words to indices."""
  print('loading word embeddings from %s' % embedding_path)
  weight_vectors = []
  word_idx = {}
  with codecs.open(embedding_path, encoding='utf-8') as f:
    for line in f:
      word, vec = line.split(u' ', 1)
      word_idx[word] = len(weight_vectors)
      weight_vectors.append(np.array(vec.split(), dtype=np.float32))
  # Annoying implementation detail; '(' and ')' are replaced by '-LRB-' and
  # '-RRB-' respectively in the parse-trees.
  word_idx[u'-LRB-'] = word_idx.pop(u'(')
  word_idx[u'-RRB-'] = word_idx.pop(u')')
  # Random embedding vector for unknown words.
  weight_vectors.append(np.random.uniform(
      -0.05, 0.05, weight_vectors[0].shape).astype(np.float32))
  print('Done !')
  return np.stack(weight_vectors), word_idx

def live_test(trained_model, data, word_idx):

    #data = "Pass the salt"
    #data_sample_list = data.split()
    live_list = []
    live_list_np = np.zeros((56,1))
    # split the sentence into its words and remove any punctuations.
    tokenizer = RegexpTokenizer(r'\w+')
    data_sample_list = tokenizer.tokenize(data)

    labels = np.array(['1','2','3','4','5','6','7','8','9','10'], dtype = "int")
    #word_idx['I']
    # get index for the live stage
    data_index = np.array([word_idx[word.lower()] if word.lower() in word_idx else 0 for word in data_sample_list])
    data_index_np = np.array(data_index)
    #print(data_index_np)

    # padded with zeros of length 56 i.e maximum length
    padded_array = np.zeros(56) # use the def maxSeqLen(training_data) function to detemine the padding length for your data
    padded_array[:data_index_np.shape[0]] = data_index_np
    data_index_np_pad = padded_array.astype(int)
    live_list.append(data_index_np_pad)
    live_list_np = np.asarray(live_list)
    type(live_list_np)

    # get score from the model
    score = trained_model.predict(live_list_np, batch_size=1, verbose=0)
    #print (score)

    single_score = np.round(np.argmax(score)/10, decimals=2) # maximum of the array i.e single band

    # weighted score of top 3 bands
    top_3_index = np.argsort(score)[0][-3:]
    top_3_scores = score[0][top_3_index]
    top_3_weights = top_3_scores/np.sum(top_3_scores)
    single_score_dot = np.round(np.dot(top_3_index, top_3_weights)/10, decimals = 2)

    #print (single_score)
    return single_score_dot

spray.head()

import utility_functions as uf

gloveFile = 'path for/glove_6B_100d.txt'
weight_matrix, word_idx = load_embeddings(gloveFile)

#Load the Model
new_model = tf.keras.models.load_model('model/best_model.hdf5')

txt = "this is a very good product 10/10 would recommend !!"
result = live_test(new_model, txt, word_idx)
print(result)

l = list(spray["Review_text"])
scores = []
indices = []

for i in range(0,len(l)):
    if(len(l[i].split(" ")) < 55):
        scores.append(live_test(new_model, l[i], word_idx))
    else:
        indices.append(i)

print(len(scores))
print(len(spray))
print(len(indices))

new = spray

new.drop(new.index[indices], inplace=True)
new = new.reset_index(drop=True)
print(new.head(20))

new['Score'] = scores
print(new.head(20))

"""## VISUALIZATION"""

import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt

#converting data to Date_time
new['Date'] = new['Date'].apply(lambda x: dt.datetime.strptime(x,'%Y-%m-%d'))

new.info()

new['day'] = new['Date'].dt.day
new['month'] = new['Date'].dt.month
new['year'] = new['Date'].dt.year

new.head(15)

temp_p = new.loc[new['Score'] > 0.55]
temp_nu = new.loc[((new['Score'] > 0.45) & (new['Score'] < 0.55))]
temp_n = new.loc[new['Score'] < 0.45]

import pandas as pd
import seaborn as sns

sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})


# Initialize the FacetGrid object
pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
g = sns.FacetGrid(new, row="month", hue="month", aspect=6.6, height=.9, palette=pal)

# Draw the densities in a few steps
g.map(sns.kdeplot, "Score",
      bw_adjust=.5, clip_on=False,
      fill=True, alpha=1, linewidth=1.5)
g.map(sns.kdeplot, "Score", clip_on=False, color="w", lw=2, bw_adjust=.5)
g.map(plt.axhline, y=0, lw=2, clip_on=False)


# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .1, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)


g.map(label, "Score")

# Set the subplots to overlap
g.fig.subplots_adjust(hspace=-.35)

# Remove axes details that don't play well with overlap
g.set_titles("")
g.set(yticks=[])
g.despine(bottom=True, left=True)

sns.barplot(x="month", y="Score", data=new, palette="Blues_d")

sns.lineplot(x="month", y="Score",data=temp_p)
sns.lineplot(x="month", y="Score",data=temp_nu)
sns.lineplot(x="month", y="Score",data=temp_n)
plt.legend(labels=['Positive', 'Neutral', 'Negative'],bbox_to_anchor=(1.05, 1), facecolor='white', loc=2, borderaxespad=0.)

import matplotlib.pyplot as plt


htmp = new[['month', 'day', 'Score']].copy()
htmp = htmp.pivot_table(index='day', columns='month', values='Score')
htmp = htmp.fillna(0)

fig, ax = plt.subplots(figsize=(8,6)) 
ax = sns.heatmap(htmp,cmap="Blues", ax = ax)
ax.invert_yaxis()

import seaborn as sns

# Plot each year's time series in its own facet
g = sns.relplot(
    data=new,
    x="day", y="Score", col="month", hue="month",
    kind="line", palette="crest", linewidth=3, zorder=5,
    col_wrap=3, height=2, aspect=2, legend=True,
)

# Iterate over each subplot to customize further
for year, ax in g.axes_dict.items():

    # Add the title as an annotation within the plot
    ax.text(.8, .85, year, transform=ax.transAxes, fontweight="bold")

    # Plot every year's time series in the background
    sns.lineplot(
        data=new, x="day", y="Score", units="month",
        estimator=None, color=".7", linewidth=1, ax=ax,
    )

# Reduce the frequency of the x axis ticks
ax.set_xticks(ax.get_xticks()[0::1])

# Tweak the supporting aspects of the plot
g.set_titles("")
g.set_axis_labels("Days", "Passengers")
g.tight_layout()

import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
comment_words = ' '
stopwords = set(STOPWORDS) 

plt.figure(figsize = (10, 8), facecolor = None) 
wordcloud2 = WordCloud().generate(' '.join(temp_p['Review_text']))
plt.imshow(wordcloud2)
plt.axis("off")
plt.show()

import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
comment_words = ' '
stopwords = set(STOPWORDS) 

plt.figure(figsize = (10, 8), facecolor = None) 
wordcloud2 = WordCloud().generate(' '.join(temp_n['Review_text']).replace("good",""))
plt.imshow(wordcloud2)
plt.axis("off")
plt.show()
