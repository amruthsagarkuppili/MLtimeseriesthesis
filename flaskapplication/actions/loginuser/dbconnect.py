from mysql.connector import connect, Error

######################################
#                                    #
# @author Amruth Sagar Kuppili       #
# @university Dalhousie University   #
# @ID B00844506                      #
#                                    #
######################################

from flaskapplication.actions.loginuser.validateuser import uservalidate

class DBconnection:
    connection = None
    cursor = None

    def __init__(self):
        # try:
            self.connection = connect( host="127.0.0.1",
                        user="root",
                        password="smabuoy@135",
                        database="smadb")
        # except Error as e:
        #     raise Exception("Database connection failed")

    def __del__(self):
        self.connection.close()

    def usercheck(self, uemail, password):
        userobj = uservalidate()
        self.cursor = self.connection.cursor()
        loginstatus = userobj.uservalidation(uemail, password, self.cursor)
        return loginstatus

    def persist(self, userdata):
        userobj = uservalidate()
        self.cursor = self.connection.cursor()
        persiststatus = userobj.userpersist(userdata, self.cursor, self.connection)
        return persiststatus

    def fetchdetails(self, uemail):
        userobj = uservalidate()
        self.cursor = self.connection.cursor()
        enpass, fetsalt = userobj.fetchdetailsuser(uemail, self.cursor)
        return enpass, fetsalt
        # persiststatus = userobj.userpersist(userdata, self.cursor, self.connection)
        # return pass


    # def establishconnection(self):
    #     try:
    #         with connect(
    #                 host="127.0.0.1",
    #                 user="root",
    #                 password="smabuoy@135",
    #                 database="smadb",
    #         ) as connection:
    #             return connection
    #     except Error as e:
    #         print(e)
    #     pass