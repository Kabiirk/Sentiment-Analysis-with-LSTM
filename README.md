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

After which we proceed to score them using VADER.


2. **STEP 2**
```
def code()
{
    put some code here;
}
```
