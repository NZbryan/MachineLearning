#!/usr/bin/python
# -*- coding: utf-8 -*-

# import re
import scrapy #导入scrapy包
from bs4 import BeautifulSoup
from scrapy.http import Request ##一个单独的request的模块，需要跟进URL的时候，需要用它
from spide_web.items import SpideWebItem ##这是我定义的需要保存的字段，（导入dingdian项目中，items文件中的DingdianItem类）



class Myspider(scrapy.Spider):

    # http://www.x23us.com/class/10_1.html
    name = 'spide_web'
    allowed_domains = ['x23us.com']
    bash_url = 'http://www.x23us.com/class/'
    bashurl = '.html'

    def start_requests(self):
        for i in range(1,11):
            url = self.bash_url + str(i) + '_1' + self.bashurl
            yield Request(url, self.parse)
        #yield Request('http://www.x23us.com/class/10_1.html', self.parse)

    def parse(self, response):
        max_num =  BeautifulSoup(response.text,'lxml').find_all('div',
                                                       class_='pagelink')[0].find_all('a')[-1].get_text()
        bashurl = str(response.url)[:-7]
        for num in range(1,int(max_num)+1):
            url = bashurl + '_' + str(num) + self.bashurl
            yield Request(url, callback=self.get_name)
        # print(response.text)

        """
            yieid Request，请求新的URL，后面跟的是回调函数，你需要哪一个函数来处理这个返回值，就调用那一个函数，
            返回值会以参数的形式传递给你所调用的函数。

        """
    def get_name(self,response):
        tds = BeautifulSoup(response.text,'lxml').find_all('tr',bgcolor='#FFFFFF')
        category = BeautifulSoup(response.text, 'lxml').find('div', class_='bdsub').find('dt').get_text().split('-')[
            0].strip()
        for td in tds:
            novelname = td.find_all('a')[1].get_text()
            novelurl = td.find_all('a')[1]['href']
            author = td.find_all('td')[-4].get_text()
            every_category = category
            yield Request(novelurl, callback=self.get_chapterurl,meta={'name':novelname,'url':novelurl,'author':author,'category':every_category})

    def get_chapterurl(self, response):
        item = SpideWebItem()
        # item['name'] = str(response.meta['name']).replace('\xa0', '')
        item['name'] = response.meta['name']
        item['novelurl'] = response.meta['url']
        item['author'] = response.meta['author']
        item['category'] = response.meta['category']
        # bash_url = BeautifulSoup(response.text, 'lxml').find('p', class_='btnlinks').find('a', class_='read')['href']
        # name_id = str(bash_url)[-6:-1].replace('/', '')
        # item['author'] = str(author).replace('/', '')
        # item['name_id'] = name_id
        yield item
        # yield Request(url=bash_url, callback=self.get_chapter, meta={'name': name})
