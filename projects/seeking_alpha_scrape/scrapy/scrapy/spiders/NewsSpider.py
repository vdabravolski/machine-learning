from scrapy.contrib.spiders import CrawlSpider, Rule
from scrapy.contrib.linkextractors.sgml import SgmlLinkExtractor
from scrapy.contrib.linkextractors import LinkExtractor
import scrapy.Requests
from scrapy.selector import HtmlXPathSelector
from scrapy import log

class SeekingAlphaSpider(CrawlSpider):
    name = 'SeekingAlphaSpider'

    rules = (
        Rule(LinkExtractor(allow=("apple")), callback='parse_item')
    )

    def start_requests(self):
        ticker = getattr(self, 'ticker', 'appl')
        url = 'http://seekingalpha.com/symbol/{0}/news'.format(ticker)
        yield scrapy.Request(url, self.parse)

    def parse_item(self, response):
        item = NewsItem()
        item['title'] = str(response.xpath('//h1[@itemprop="headline"]/text()').extract())
        item['content'] = str(response.xpath('//div[@itemprop="articleBody"]/div[1]/p/text()').extract())
        item['url'] = response.urscrapyl