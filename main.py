import streamlit as st
from streamlit_option_menu import option_menu
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import joblib
import pandas as pd
import altair as alt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load text emotion detection model
pipe_lr = joblib.load(open("text/model/text_emotion_model.pkl", "rb"))

# Load image emotion recognition model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
model.load_weights('image/model.h5')

# Emotion dictionary for image recognition
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Emojis corresponding to emotions
emotions_emoji_dict = {
    "anger": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "joy": "😄",
    "neutral": "😐",
    "sadness": "😢",
    "shame" : "😢",
    "Surprised": "😲"
}

emotions_emoji_dict_img = {
    "Angry": "😠",
    "Disgusted": "🤢",
    "Fearful": "😨",
    "Happy": "😄",
    "Neutral": "😐",
    "Sad": "😢",
    "Surprised": "😲"
}


def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]


def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results


def emotion_recog(frame):
    detected_emotion = None
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier('image/haarcascade_frontalface_default.xml').detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 255), 3)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        detected_emotion = emotion_dict[maxindex]
        cv2.putText(frame, emotion_dict[maxindex], (x+20, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return frame, detected_emotion

def local_css(file_name):
    with open(file_name,'r') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

def main():
    with st.sidebar:
        selected =option_menu(
            menu_title=None,
            options=["TEXT", "IMAGE", "CONTACT"],
            icons=["cursor-text","card-image","person-lines-fill"],
            default_index=0,
        )
    
    # Text Emotion Detection
    
    if selected=="TEXT":
        st.header("Text Emotion Detection")

        with st.form(key='text_emotion_form'):
            raw_text = st.text_area("**Type Here**")
            submit_text = st.form_submit_button(label='**Submit**')

        if submit_text:
            col1, col2 = st.columns(2)
            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

            with col1:
                st.success("**Original Text**")
                st.write(raw_text)

                st.success("**Prediction**")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write("{}:{}".format(prediction, emoji_icon))
                st.write("**Confidence:{}**".format(np.max(probability)*100),"%")

            with col2:
                st.success("**Prediction Probability**")
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions", "probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
                st.altair_chart(fig, use_container_width=True)
    
    if selected == "IMAGE":
    # Image Emotion Recognition
        st.header("Image Emotion Recognition")
        uploaded_file = st.file_uploader("**Choose an image...**", type=['jpg', 'png'])

        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)
            result, predicted_emotion = emotion_recog(frame)

            st.success("**Uploaded Image**")
            st.image(result, caption='**Uploaded Image**', use_column_width=True)

            st.success("**Prediction**")
            emoji_icon = emotions_emoji_dict_img[predicted_emotion]
            st.write("{}:{}".format(predicted_emotion, emoji_icon))

    if selected == "CONTACT":
        st.header("**Contributors**")
        contributors = {
            "ANSH VARSHNEY": {"email": "anshvarshney3@gmail.com", 
                              "github_link": "https://github.com/ansh0707",
                              "details":{
                                  "Reg No" : "12006893",
                                  "University":"Lovely Professional University"
                              }},
            "SARA BORA": {"email": "sarabora420@gmail.com", 
                          "github_link": "https://github.com/sara-bora",
                          "details":{
                              "Reg No" : "12013194",
                              "University":"Lovely Professional University"
                          }},
            "SARTHAK MISHRA": {"email": "sam4sarthak@gmail.com",
                                "github_link": "https://github.com/SarthakMishra0307",
                                "details":{
                                    "Reg No" : "12018433",
                                    "University":"Lovely Professional University"
                                }},
            "SATYAM DUBEY": {"email": "satyamdubey2988@gmail.com",
                             "github_link": "https://github.com/dubeysatyam2002",
                             "details":{
                                    "Reg No" : "12014267",
                                    "University":"Lovely Professional University"
                            }},
        }

        for name, info in contributors.items():
            st.markdown(
                '''
                <style>
                div[data-testid="stExpander"] details div[data-testid="stExpanderContent"] summary {
                    font-size: 1.2rem;
                    color: blue;
                    /* Add any other styles you want */
                }
                </style>
                ''',
                unsafe_allow_html=True
            )
            
            with st.expander(name):
                
                st.write(f"**Email:** {info['email']}",unsafe_allow_html=True)
                st.write(f"**GitHub:** [{name}]({info['github_link']})")
                st.write(f"**Reg No:** {info['details']['Reg No']}")
                st.write(f"**University:** {info['details']['University']}")
                #st.markdown('</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
