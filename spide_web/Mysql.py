import MySQLdb
from spide_web import settings

MYSQL_HOSTS = settings.MYSQL_HOSTS
MYSQL_USER = settings.MYSQL_USER
MYSQL_PASSWORD = settings.MYSQL_PASSWORD
MYSQL_PORT = settings.MYSQL_PORT
MYSQL_DB = settings.MYSQL_DB

cnx = MySQLdb.connect(host=MYSQL_HOSTS,user=MYSQL_USER, passwd=MYSQL_PASSWORD, db=MYSQL_DB,charset='utf8')
cur = cnx.cursor()

class Sql:

    @classmethod
    def insert_dd_name(cls, xs_name, xs_author, category):
        sql = 'INSERT INTO dd_name (`xs_name`, `xs_author`, `category`) VALUES (%(xs_name)s, %(xs_author)s, %(category)s)'
        value = {
            'xs_name': xs_name,
            'xs_author': xs_author,
            'category': category,
            # 'name_id': name_id
        }
        cur.execute(sql, value)
        cnx.commit()


    @classmethod
    def select_name(cls, name):
        sql = "SELECT EXISTS(SELECT 1 FROM dd_name WHERE xs_name=%(name)s)"
        value = {
            'name': name
        }
        cur.execute(sql, value)
        for name_id in cur:
            return name_id[0]
