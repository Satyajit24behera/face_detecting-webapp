import cv2
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from PIL import Image


class FaceDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier("C:\\Users\\Satyajit\\OneDrive\\Desktop\\raw.githubusercontent.com_opencv_opencv_master_data_haarcascades_haarcascade_frontalface_default.xml")

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

        return img
    


def main():
 
    image = Image.open("C:\\Users\\Satyajit\\OneDrive\\Desktop\\logo.png")

    # Display the image using Streamlit
    st.image(image, use_column_width=50)
    st.title("Real-time Face Detection")
     
    webrtc_ctx = webrtc_streamer(key="example", video_transformer_factory=FaceDetectionTransformer)


    return

    

    if st.button("Start"):
        if webrtc_ctx.state.playing:
            st.warning("The video stream is already running.")
        else:
            webrtc_ctx.play()
            st.success("Video stream started.")
    else:
        if webrtc_ctx.state.playing:
            webrtc_ctx.stop()
            st.warning("Video stream stopped.")
        else:
            st.warning("The video stream is already stopped.")


if __name__ == "__main__":
    main()
