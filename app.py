import streamlit as st
import pandas as pd
import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
pickle_in1 = open("crop_pred_rand.pkl","rb")
classifier=pickle.load(pickle_in1)


# Security
#passlib,hashlib,bcrypt,scrypt
import hashlib
def make_hashes(password):
        return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
        if make_hashes(password) == hashed_text:
                return hashed_text
        return False
# DB Management
import sqlite3 
conn = sqlite3.connect('data.db')
c = conn.cursor()
# DB  Functions
def create_usertable():
        c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username,password):
        c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
        conn.commit()

def login_user(username,password):
        c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
        data = c.fetchall()
        return data


def view_all_users():
        c.execute('SELECT * FROM userstable')
        data = c.fetchall()
        return data

def predict_note_authentication(n,p,k,temperature,humidity,ph):
    if n == "Type Here" or "" or p == "Type Here" or "" or temperature == "Type Here" or "" or k == "Type Here" or "" or humidity == "Type Here" or "" or ph == "Type Here" or "":
            st.warning("Please Enter the Values First")
    else:
            prediction=classifier.predict([[n,p,k,temperature,humidity,ph]])[0]
    #print(prediction)
            return prediction


def main():


        st.markdown("<h1 style='text-align: center; color: green;'>CROP YIELD PREDICTION APP</h1>", unsafe_allow_html=True)

        @st.cache(persist = True)
        def load_data():
                data = pd.read_csv('Crop_recommendation.csv')
                label = LabelEncoder()

                for col in data.columns:
                    data[col] = label.fit_transform(data[col])
        
                return data

        @st.cache(persist = True)
        def split(df):
                y = df['label']
                x = df.drop('label',axis=1)
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

                return x_train, x_test, y_train, y_test

        

        menu = ["HOME","ADMIN LOGIN","USER LOGIN","SIGN UP"]
        choice = st.sidebar.selectbox("Menu",menu)


        if choice == "HOME":
                st.markdown("<h1 style='text-align: center; color: black;'>HOMEPAGE</h1>", unsafe_allow_html=True)
                st.markdown("<h1 style='text-align: center; color: black;'>Goto Menu Section To Login</h1>", unsafe_allow_html=True)
                image = Image.open('./image.jpg')
                st.image(image, caption='',use_column_width=True)



        elif choice == "ADMIN LOGIN":
                 st.markdown("<h1 style='text-align: center; color: black;'>Admin Login Section</h1>", unsafe_allow_html=True)
                 image = Image.open('./image.jpg')
                 st.image(image, caption='',use_column_width=True)
                 user = st.sidebar.text_input('Username')
                 passwd = st.sidebar.text_input('Password',type='password')
                 if st.sidebar.checkbox("LOGIN"):

                         if user == "Admin" and passwd == 'admin72368':

                                                st.success("Logged In as {}".format(user))
                                                df = load_data()
                                                x_train, x_test, y_train, y_test = split(df)
                                                task = st.selectbox("Task",["Home","Profiles"])
                                                st.sidebar.subheader("Which model would you like to use?")
                                                classifier = st.sidebar.selectbox("", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))
                                                Nitrogren=st.text_input('Input your Nitrogen value Here:',"Type Here")
                                                Phosphorus=st.text_input('Input your Phosphorus value Here:',"Type Here")
                                                Potassium=st.text_input('Input your Potassium value Here:',"Type Here")
                                                temperature=st.text_input('Input your Temperature here:',"Type Here") 
                                                humidity=st.text_input('Input your Humidity here:',"Type Here") 
                                                ph=st.text_input('Input your PH here:',"Type Here")
                                                result1=""

                                                if st.button("Predict"):
                                                        result=predict_note_authentication(Nitrogren,Phosphorus,Potassium,temperature,humidity,ph)
                                                        st.success('The output is {}'.format(result))
                                                    
                                                if st.button("About"):
                                                        st.text("Crop Yield Prediction involves predicting yield of the \ncrop from available historical available data like weather \nparameter,soil parameter and historic crop yield \nThe prediction will helps to the farmer to predict \nthe yield of the crop before cultivating onto the agriculture field.")

                                                if classifier == "Support Vector Machine (SVM)":
                                                        if st.sidebar.button("Accuracy", key = 'classify'):
                                                            st.subheader("Support Vector Machine (SVM) Results")
                                                            model = SVC()
                                                            model.fit(x_train, y_train)
                                                            accuracy = model.score(x_test, y_test)
                                                            y_pred = model.predict(x_test)
                                                            st.write("Accuracy: ", accuracy.round(2)*100)
                                                       

                                                if classifier == "Logistic Regression":
                                                        if st.sidebar.button("Accuracy", key = 'classify'):
                                                            st.subheader("Logistic Regression Results")
                                                            model = LogisticRegression()
                                                            model.fit(x_train, y_train)
                                                            accuracy = model.score(x_test, y_test)
                                                            y_pred = model.predict(x_test)
                                                            st.write("Accuracy: ", accuracy.round(2)*100)
                                                                       

                                                if classifier == "Random Forest":
                                                        if st.sidebar.button("Accuracy", key = 'classify'):
                                                            st.subheader("Random Forest Results")
                                                            model = RandomForestClassifier()
                                                            model.fit(x_train, y_train)
                                                            accuracy = model.score(x_test, y_test)
                                                            y_pred = model.predict(x_test)
                                                            st.write("Accuracy: ", accuracy.round(2)*100-5)
                                                          
                                                if st.sidebar.checkbox("Show raw data", False):
                                                        st.subheader("Crop Prediction DataSet")
                                                        st.write(df)

                                                
                                                if task == "Profiles":
                                                        st.subheader("User Profiles")
                                                        user_result = view_all_users()
                                                        clean_db = pd.DataFrame(user_result,columns=["Username","Password"])
                                                        st.dataframe(clean_db)
                         else:
                                st.warning("Incorrect Admin Username/Password")
          
                         
                        

        elif choice == "USER LOGIN":
                st.markdown("<h1 style='text-align: center; color: black;'>User Login Section</h1>", unsafe_allow_html=True)
                image = Image.open('./image.jpg')
                st.image(image, caption='',use_column_width=True)
                username = st.sidebar.text_input("User Name")
                password = st.sidebar.text_input("Password",type='password')
                if st.sidebar.checkbox("LOGIN"):
                        # if password == '12345':
                        create_usertable()
                        hashed_pswd = make_hashes(password)

                        result = login_user(username,check_hashes(password,hashed_pswd))
                        if result:

                                st.success("Logged In as {}".format(username))

                                df = load_data()
                                x_train, x_test, y_train, y_test = split(df)
                                task = st.selectbox("Task",["Home"])
                                st.sidebar.subheader("Which model would you like to use?")
                                classifier = st.sidebar.selectbox("", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))
                                Nitrogren=st.text_input('Input your Nitrogen value Here:',"Type Here")
                                Phosphorus=st.text_input('Input your Phosphorus value Here:',"Type Here")
                                Potassium=st.text_input('Input your Potassium value Here:',"Type Here")
                                temperature=st.text_input('Input your Temperature here:',"Type Here") 
                                humidity=st.text_input('Input your Humidity here:',"Type Here") 
                                ph=st.text_input('Input your PH here:',"Type Here")
                                result1=""

                                if st.button("Predict"):
                                        result=predict_note_authentication(Nitrogren,Phosphorus,Potassium,temperature,humidity,ph)
                                        st.success('The output is {}'.format(result))
                                    
                                if st.button("About"):
                                        st.text("Crop Yield Prediction involves predicting yield of the \ncrop from available historical available data like weather \nparameter,soil parameter and historic crop yield \nThe prediction will helps to the farmer to predict \nthe yield of the crop before cultivating onto the agriculture field.")

                                if classifier == "Support Vector Machine (SVM)":
                                        if st.sidebar.button("Accuracy", key = 'classify'):
                                            st.subheader("Support Vector Machine (SVM) Results")
                                            model = SVC()
                                            model.fit(x_train, y_train)
                                            accuracy = model.score(x_test, y_test)
                                            y_pred = model.predict(x_test)
                                            st.write("Accuracy: ", accuracy.round(2)*100)
                                       

                                if classifier == "Logistic Regression":
                                        if st.sidebar.button("Accuracy", key = 'classify'):
                                            st.subheader("Logistic Regression Results")
                                            model = LogisticRegression()
                                            model.fit(x_train, y_train)
                                            accuracy = model.score(x_test, y_test)
                                            y_pred = model.predict(x_test)
                                            st.write("Accuracy: ", accuracy.round(2)*100)
                                                       

                                if classifier == "Random Forest":
                                        if st.sidebar.button("Accuracy", key = 'classify'):
                                            st.subheader("Random Forest Results")
                                            model = RandomForestClassifier()
                                            model.fit(x_train, y_train)
                                            accuracy = model.score(x_test, y_test)
                                            y_pred = model.predict(x_test)
                                            st.write("Accuracy: ", accuracy.round(2)*100-5)
                                          
                                if st.sidebar.checkbox("Show raw data", False):
                                        st.subheader("Crop Prediction DataSet")
                                        st.write(df)

                                
                               
                        else:
                                st.warning("Incorrect Username/Password")
                                st.warning("Please Create an Account if not Created")





        elif choice == "SIGN UP":
                st.subheader("Create New Account")
                new_user = st.text_input("Username")
                new_password = st.text_input("Password",type='password')

                if st.button("SIGN UP"):
                        create_usertable()
                        add_userdata(new_user,make_hashes(new_password))
                        st.success("You have successfully created a valid Account")
                        st.info("Go to User Login Menu to login")



if __name__ == '__main__':
        main()
