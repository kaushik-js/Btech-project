from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
from flask import redirect
import speech_recognition as sr
import randomTextGen as rt
import test2 as tt
import mongopy as mpy
from bson import Binary
from secure import secure
import json

r = sr.Recognizer()
app = Flask(__name__)

randtext = ''
name_file = ''
login_status = ''
user_email = ''
user_password = ''

udata = dict()
logindt = dict()

def checkText(text):
    global randtext
    text = text.replace('  ',',')
    text = text.replace(' ',',')
    randtext = randtext.replace('  ',',')
    randtext = randtext.replace(' ',',')


    if randtext[0]==',':
        randtext = randtext[1:]
    print(text, randtext)
    if text == randtext:
        return True
    else:    
        return False
    
    
def checkVoice():

    if tt.VoiceAuth().loginSuccess(name_file):
       return True
    else:
        return False
    

@app.route("/")
def index():
    global randtext
    randtext = rt.randText().getText()
    return render_template('index.html',text = randtext)


@app.route("/login")
def login():
    global randtext
    randtext = rt.randText().getText()
    return render_template('loginPage.html',text = randtext)

@app.route("/register")
def register():
    global randtext
    randtext = rt.randText().getText()
    return render_template('registerPage.html',text = randtext)

@app.route('/result', methods=['POST'])
def result():
    if 'data' in request.files:
        file = request.files['data']
        type = request.form['action']
        global userdata
        global name_file
        global object_id
        name_file = request.form['username']
        name_file = (name_file.split('@'))[0]
        global user_name
        user_name = name_file    

        name_file = secure_filename(name_file)
        global logintext
        logintext = type
        if type == 'login':
            name_file += ".flac"
            name_file = "test/" + name_file
            file.save(name_file)
            global user_email
            global user_password
            user_email = request.form['username']
            user_password = request.form['password']            
            
        else:
            name_file += ".flac"
            name_file = "train/" + name_file
            file.save(name_file)
            tt.VoiceAuth().register_user(name_file)
            with open(name_file,'rb') as f:
                audio_bytes = f.read()          
            fl = request.files['image']
            global udata
            udata = {
                'name':request.form['name'],
                'pro_id':[],
                'emails':request.form['username'],
                'mobile' : request.form['mobile'],
                'college': request.form['college'],
                'about': request.form['about'],
                'dept': request.form['dept'],
                'image': Binary(fl.read()),
                'password': secure().encode(request.form['password']),
                'audioFile': Binary(audio_bytes),
                'isAuth': False
            }                 
    
    return 'Success'

@app.route("/audioTotext")
def audioTotext():
    global name_file
    print("file name : "+name_file)
    with sr.AudioFile(name_file) as source:
            audio_data = r.record(source)
            global text
            text = r.recognize_google(audio_data)
    text = text.lower()
    ct = checkText(text)
    if logintext !=  'login':
        if ct:            
            global udata
            mpy.mongodb().insertData(udata)
            print(udata)
            return render_template("registersuc.html",head = user_name + " registered successfully")
        else:
            return render_template("loginfail.html",head='\nText : '+str(ct),text = text)

    else:
        
        vt = checkVoice()
        global login_status
        global user_email
        global user_password
        global user_data
        user_data =  mpy.mongodb().getDataByEmail(user_email)
        if user_data == None:
            login_status = 'No user found'
            return render_template("loginfail.html",head=login_status+'\n',text = '')
                
        else :
            if secure().encode(user_password) == user_data['password']:
                login_status = 'True'
                if user_data['isAuth'] == 'False':
                    mpy.mongodb().updateAuth(user_email)               
                if ct and vt :
                    return redirect('http://localhost:8000')
                else:
                    return render_template("loginfail.html",head=login_status+'\n'+'Text : '+str(ct)+'\nVoice Authentication : '+str(vt),text = text)
                
            else:
                login_status = 'Wrong Password'
                return render_template("loginfail.html",head=login_status+'\n'+'Text : '+str(ct)+'\nVoice Authentication : '+str(vt),text = text)
                


@app.route('/getUser')
def getUser():
    global user_email
    getuser = mpy.mongodb().getDataByEmail(user_email)
    del getuser['image']
    del getuser['_id']
    del getuser['audioFile']
    return json.dumps([getuser])

if __name__ == '__main__':
    app.run(debug=True)