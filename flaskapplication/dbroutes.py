from flask import render_template, Blueprint, request, flash
from flaskapplication.actions.loginuser.dbconnect import DBconnection
from flaskapplication.actions.loginuser.cryptopass import passcrypt
from mysql.connector import  Error

# Blueprint Configuration
db_bp = Blueprint(
    'db_bp', __name__
)

@db_bp.route("/login", methods=['POST'])
def login():
    status = False
    uemail = request.form.get('loginemail')
    pwd = request.form.get('loginpass')
    try:
        dbobj = DBconnection()
        enpass, fetsalt = dbobj.fetchdetails(uemail)
        decryptobj = passcrypt()
        decryptedpass = decryptobj.passworddecrypt(fetsalt, enpass)
        status = pwd == decryptedpass
        # status = dbobj.usercheck(uemail, pwd)
        flash("Login Successful, now started training....") if status else flash("Failed to Login")
        del dbobj
    except Error as error:
        flash("Check Database Connection ")

    return render_template('index.html', loginsuccessstatus=status)


@db_bp.route("/adduser", methods=['POST'])
def adduser():
    userdata = {}
    status = False
    userdata['newuemail'] = request.form.get('addnewemail')
    userdata['newpwd'] = request.form.get('addnewpass')
    exemail = request.form.get('addexemail')
    expass = request.form.get('addexpass')
    try:
        dbobj = DBconnection()
        enpass, fetsalt = dbobj.fetchdetails(exemail)
        decryptobj = passcrypt()
        decryptedpass = decryptobj.passworddecrypt(fetsalt, enpass)
        status = expass == decryptedpass
        # status = dbobj.usercheck(exemail, expass)
        if status:
            encryptobj = passcrypt()
            userdata['encryptedpass'], userdata['salt'] = encryptobj.passwordencrypt(userdata['newpwd'])
            persiststatus = dbobj.persist(userdata)
            flash("successfully added") if persiststatus else flash("Failed to Add")

        del dbobj
    except Error as error:
        flash(error)
    return render_template('index.html')