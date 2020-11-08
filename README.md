# **SENTIMENT ANALYSIS WITH LSTM**
In today’s world, our exponential increase of exposure to new and ever-evolving information has had significant impact on our growth as a society. The internet and shift of mainstream media to internet platforms has only amplified this evolution, to such an extent that a lot of decisions that we make in our daily lives are extensively beguiled by what we see and hear on the internet. For example, before buying a product, we’ll always check for other’s reviews and opinions about that product on the internet (the whole culture or ‘unboxing/review videos’ started from this) and then proceed to buy it, sometimes even at the expense of our own objective analysis. The advantages are many, which benefit the customers and the product companies alike. For example, a customer can look up a product review online and get an ‘pseudo-immersive’ experience of the product, as if he/she himself used it, this allows the customer to make an informed decision. For the companies, they can employ methods like sentiment analysis and semantic searches to see what the consumers want and can make new products, based on those insights, to cater to a specific or a wider consumer-base.
  
Of course, these data-driven insights aren’t generated out of thin air, often, thousands of factors and data-points need to be factored in and an appropriate analysis need to be applied for an in-depth analysis which can help create business strategies. This is where Sentiment analysis comes in, it helps the companies get a pulse on what the consumers are thinking about their products and gauge public mood, which in-turn helps them take well-informed and deeply consumer-centric decisions. This project aims at using LSTM-based neural network to analyse the sentiments related to a product to gauge the relevant emotions displayed by the public with regards to the product.


## Contents
* Introduction
* Tools used
* Get Started !

### Introduction
In today’s world, our decision-making is centred around other people’s opinion about a certain product or service. It is here that NLP methodologies like Sentiment analysis shine the most as they allow people a good idea about the public opinion on the product and make an informed decision based off it. 
As compared to RNNs, LSTMs don’t have the problem of vanishing gradient that often and their models are also lightweight and perform well, making LSTM-based models ideal for mobile computation or usage in machines with less resources. Not only are LSTMs faster, they are more accurate as well for NLP-related tasks. This project aims at utilising those advantages to bring accurate insights to the consumer for a low computational cost.

---

### Tools Used
* Google Collaboratory Python Notebooks (Running intensive Models, Iterative testing etc.)
* VS Code (For python scripts)
* Pandas (Dataframe Operations)
* Scrapy (Web Scraping)
* Kera/Tensorflow (Sentiment Analysis)
* Seaborn/Matplotlib (Visualization)
* [VADER (Valence Aware Dictionary and sEntiment Reasoner)](https://github.com/cjhutto/vaderSentiment) (Preparing Training Data)

---

#### NOTE: There are 2 versions of the Sentiment Analysis Model Training and Result Visualization files, a notebook (.ipynb) or python (.py) scripts to be run locally use whichever is suited, However, The notebooks are more convenient as they only need to be uploaded to your collaboratory instance

### Get started !
1. **STEP 1: Web Scraping**

**NOTE:** Make sure to have [Scrapy installed](https://docs.scrapy.org/en/latest/intro/install.html) first before running these commands.
 
First, we’ll be using SCRAPY to extract Reviews and their metadata like number of stars, date the review was posted on etc. This is done via DOM traversal in scrapy where the Spider coded by us goes through the HTML DOM of the page upto the tag which stores the required data and save it in a dataframe. This can be done by navigating to the **Amazon Web Scraper**(The folder which has scrapy.cfg) via your CLI & typing the following:

```
scrapy crawl Reviews -o filename.csv (or .json or.xml)
```

Now the data will be outputed in the .csv in the directory, the data extracted would look something like this:
 
> **“\n\n\n \n\nThis is a very good productÅÇ 10/10 would recommend æç êÿ \t\t!!”**
 
Which is near unusable in this state, we would need to perform some initial cleaning and data sanitation like removing whitespaces, special characters, making every sentence lowercase, converting text in Date column to Datetime data type etc. before we can move on to allotting them scores indicating their degree of positivity, negativity or Neutrality to end up with something like this:
 
> **“this is a very good product 10/10 would recommend !!”**
 
**NOTE:** This is done with data_sanitation.py, it does most of the cleaning and outputs a *filename_sanitized.csv* which is used by the *sentiment.py* to score. However , do modify if some more functionality is required.
 
After which we proceed to score them using VADER.
 
2. **STEP 2: Scoring the Training Set using VADER**
Correct the filepaths as required and then just run the *sentiment.py* Python Script, it should output a new *filename_sanitized_sentiment.csv*
 
**NOTE:** Scraping can be done again as required if data obtained in the first scrape wasn't enough (Which was the case here, 3 files worked for me). it is advised to Scrape multiple files first and then score them in one go after modifying *sentiment.py*.
 
3. **STEP 3: Preprocessing of Text**
Before we put our text into the LSTM for training, we need to understand that we need to feed it in a form the network can understand. LSTMs and even other ANNs can’t comprehend words “as is” therefore they need to be converted into sequences/vectors and then into embedding before being fed into the network.
 
This is done in the *(Training)Sentiment_Analysis_using_LSTMs.ipynb* or the *sentiment_analysis_using_lstms.py*.
 
4. **STEP 4: Creation of LSTM and Training**
```
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(embedding_dim, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])
```
Here we create a sequential model using Sequential() object from Keras. First we’ll be adding the embedding layer which takes the our created word vectors as inputs and sends them to the LSTM layer which we can create using model.add(LSTM(n)) method, where ‘n’ is the number of LSTM units in our layer. If we want to make Multi-layered LSTM models, we’ll need to put the return_sequence parameter to true.
And we finally train our model with:
```
model.fit(train_padded, training_label_seq, epochs=num_epochs, validation_data=(validation_padded, validation_label_seq), verbose=2)
```
 
4. **STEP 5: Predicting compound scores on our Test Set using the model and Visualising the Results**
For this step we’ll be using our scraped dataset which we did not score with vader. This can be done with the help of the model.predict() method, but we’ll still need to convert each of our reviews into vectors with the help of tokenizer.text_to_sequence() method and then we pad the sequence with pad_sequences() method before putting it into the model.predict() function.

The Final part of this project would be to visualize our predicted values, because in a real-world setting, most people won’t be able to gain insights just by a table, that too a table having 1000’s of rows. Depicting that data via easy-to-understand charts and graphs makes it easier for the other person to infer the implications of the data like various trends of certain emotions and the rise and fall of these emotions in certain time periods.
```
s = "this is a very good product 10/10 would recommend !!"
instance = tokenizer.texts_to_sequences(s)
instance = pad_sequences(instance, padding='post', maxlen=100)
res = live_test(new_model, txt, word_idx)
print(res)
```
This is usually done with just ```model.predict()``` method, but separate function ```live_test()``` has been used so as to give a more calibrated sentiment score. it is defined in the *(Testing_+_Visualization)Sentiment_Analysis_using_LSTMs.ipynb* or *visualization.py* & will be done with the Visualization itself.

# All Done ! 
