# Emotion Detection from Text & Images

This project implements emotion detection from both text and images using machine learning models. It consists of three web pages where users can interact with the application to analyze emotions:

- **[Text Emotion Detection](#text-emotion-detection)**: Users can input text, and the application predicts the emotion conveyed by the text.
- **[Face Emotion Detection](#face-emotion-detection)**: Users can upload images containing faces, and the application predicts the emotion expressed by each face.
- **[Contributors](#contributors)**: Details of the contributors involved in the project are provided.

## Deployed Application

The application is deployed and accessible through the following link: [Emotion Detection Web App](https://emotion-detection-k20uw.streamlit.app/)

## Text Emotion Detection

In the Text Emotion Detection page, users can input text into a text area. The application then predicts the emotion conveyed by the text. Emotions predicted include anger, disgust, fear, joy, neutral, sadness, and surprise. Predictions are displayed along with confidence scores.

## Face Emotion Detection

In the Face Emotion Detection page, users can upload images containing faces. The application detects faces in the image and predicts the emotion expressed by each face. Emotions predicted include anger, disgust, fear, joy, neutral, sadness, and surprise. Predictions are displayed alongside the uploaded image.

## Contributors

The project was developed by the following contributors:

- **Ansh Varshney**
  - Email: anshvarshney3@gmail.com
  - GitHub: [ansh0707](https://github.com/ansh0707)
  - Registration No: 12006893
  - University: Lovely Professional University

- **Sara Bora**
  - Email: sarabora420@gmail.com
  - GitHub: [sara-bora](https://github.com/sara-bora)
  - Registration No: 12013194
  - University: Lovely Professional University

- **Sarthak Mishra**
  - Email: sam4sarthak@gmail.com
  - GitHub: [SarthakMishra0307](https://github.com/SarthakMishra0307)
  - Registration No: 12018433
  - University: Lovely Professional University

- **Satyam Dubey**
  - Email: satyamdubey2988@gmail.com
  - GitHub: [dubeysatyam2002](https://github.com/dubeysatyam2002)
  - Registration No: 12014267
  - University: Lovely Professional University

## Usage

1. Clone the repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the Streamlit application using `streamlit run app.py`.
4. Access the web application through the provided URL.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- The image emotion recognition model was trained using the FER2013 dataset.
- The text emotion detection model was trained on various text datasets.
- Special thanks to the Streamlit team for providing an easy-to-use platform for building web applications in Python.

## Note

This application is for educational and demonstration purposes only. The accuracy of emotion detection may vary depending on input data and model performance.
