from mysql.connector import  Error
class uservalidate:

    def fetchdetailsuser(self, uemail, cursor):
        query =  "select enpass, salt from encryptedusers where email = '{}'".format(uemail)
        cursor.execute(query)
        results = cursor.fetchall()
        encryptedpass = (results[0])[0]
        fetchedsalt = (results[0])[1]
        return encryptedpass, fetchedsalt

    def uservalidation(self, uemail, pwd, cursor):
        query =  "select EXISTS( select * from users where uemail = '{}' and password = '{}')".format(uemail,pwd)
        cursor.execute(query)
        results = cursor.fetchall()
        status =  (results[0])[0]
        return True if status==1 else False

    def userpersist(self, userdata, cursor, connection):
        query = "insert into encryptedusers(email, enpass, salt) values ('{}','{}','{}')".format(userdata['newuemail'], userdata['encryptedpass'], userdata['salt'])
        try:
            cursor.execute(query)
            connection.commit()
        except Error as error:
            connection.rollback()
            print(error)
            return False
        return True


