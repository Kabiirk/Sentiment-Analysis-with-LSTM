import scrapy
import re
import datetime
from ..items import AmazonReviewItem

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

class ReviewSpider(scrapy.Spider):
    name = "Reviews"
    start_urls = [
        #'https://www.amazon.in/Amazon-Echo-Smart-speaker-Powered/product-reviews/B0749YXL1J/ref=cm_cr_arp_d_paging_btm_next_1?ie=UTF8&reviewerType=all_reviews&pageNumber=1'
        #'https://www.amazon.in/Echo-Dot-3rd-Gen-improved/product-reviews/B0792HC9RP/ref=cm_cr_arp_d_paging_btm_next_1?ie=UTF8&reviewerType=all_reviews&pageNumber=1'
        #'https://www.amazon.in/Lifebuoy-Antibacterial-Germ-Kill-Spray/product-reviews/B08779GB46/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews',
        #'https://www.amazon.in/Lifebuoy-Alcohol-Based-Protection-Sanitizer/product-reviews/B0866JTZXN/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews',
        #"https://www.amazon.in/Dummy-July-July196/product-reviews/B07VN7T85T/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews",
        "https://www.amazon.in/Lifebuoy-Antibacterial-Germ-Kill-Spray/product-reviews/B08GGDMWM2/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews",
    
    ]
    
    def parse(self, response):
        #variable declaration
        items = AmazonReviewItem()
        data = response.css('#cm_cr-review_list')
        review_title = data.css('.a-link-normal.review-title').extract()
        review_text = data.css('.review-text').extract()
        star_rating = data.css('.review-rating').xpath('.//text()').extract()
        date = data.css('.review-date').xpath('.//text()').extract()
        count = 0
        n = len(review_text)

        '''
        for count in range(len(review)):
            yield{"Titleeeeeeeeeeeeeeeeeee":cleanhtml(review[count])}
            yield{"Starsssssssssssssssssss":int(star_rating[count][0])}
            yield{"Dateeeeeeeeeeeeeeeeeeee":date[count]}
        '''

        #return all reviews on a page
        for count in range(n):
            items['Review_title'] = cleanhtml(review_title[count])
            items['Review_text'] = cleanhtml(review_text[count])
            items['Stars'] = int(star_rating[count][0])
            items['Date'] = date[count]

            yield items

        #Scrape multiple pages for a given product
        next_page = response.css('li.a-last a::attr(href)').get()
        if next_page is not None:
            yield response.follow(next_page, callback= self.parse)

'''
IMPORTANT CLI COMMANDS:

scrapy crawl Reviews  ==> runs the crawler with the given URL
scrapy crawl Reviews -o filename.csv(or .json or.xml)  ==> exports the scrapped data into required file format
scrapy shell ==> opens the scrapy shell for prototyping and DOM traversal
'''