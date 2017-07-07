#!/usr/bin/python
# -*- coding: utf-8 -*-

from spide_web.Mysql import Sql
from spide_web.items import SpideWebItem


class SpideWebPipeline(object):

    def process_item(self, item, spider):
        #deferToThread(self._process_item, item, spider)
        if isinstance(item, SpideWebItem):
            name = item['name']
            ret = Sql.select_name(name)
            if ret == 1:
                print('已经存在了')
                pass
            else:
                xs_name = item['name']
                xs_author = item['author']
                category = item['category']
                Sql.insert_dd_name(xs_name, xs_author, category)
                print('开始存小说标题')
