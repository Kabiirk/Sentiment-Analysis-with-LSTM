# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html
#ITEM CONTAINER

import scrapy


class AmazonReviewItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    Review_title = scrapy.Field()
    Review_text = scrapy.Field()
    Stars = scrapy.Field()
    Date = scrapy.Field()