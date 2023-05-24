import pymongo
from bson.binary import Binary
from pymongo import MongoClient


connection_url = 'mongodb+srv://kosec:2577@cluster0.hfzw1pi.mongodb.net/?retryWrites=true&w=majority'

class mongodb:
    def getConnection(self):
        global connection_url
        return pymongo.MongoClient(connection_url)

    
    def insertData(self,udata):
        client = mongodb().getConnection()
        collection = client['userData']
        user_tb = collection['users']
        res = user_tb.insert_one(udata)
        return
    
    def updateAuth(self,email):
        client = mongodb().getConnection()
        collection = client['userData']
        user_tb = collection['users']
        if (mongodb().getDataByEmail(email)['isAuth']) == False:
            user_tb.update_one({'emails':email},{'$set':{'isAuth':True}})
        else:
            user_tb.update_one({'emails':email},{'$set':{'isAuth':False}})
        return mongodb().getDataByEmail(email)['isAuth']


    def getDataByEmail(self,email):
        client = mongodb().getConnection()
        collection = client['userData']
        user_tb = collection['users']
        return user_tb.find_one({'emails':email})

