# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy
from scrapy.item import Item, Field


class NewsItem(scrapy.Item):
    # define the fields for your item here like:
    title = Field()
    date = Field()
    url = Field()
    content = Field()
    company = Field()