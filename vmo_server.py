"VMO stands for Virtual Mike Online. The server components are abstracted from the actual SadTalker stream"

from flask import Flask, Response, render_template
import cv2
import time
import generate_stream


## Accept input of an audio stream
## Render video based on the audio stream and the picture
## sync the audio stream via the still reading functionality
## still_reading/load_and_stream video comes from generate()
app = Flask(__name__)


# Load and stream video frames
def load_and_stream_video(pic_path, audio_path):
    frame_delay = 1 / 25
    for image in generate_stream.avatar_stream_generator(pic_path, audio_path):
        if image is not None and image.size > 0:
            print(image)
        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode(".jpg", image)
        if ret:
            print("Success")
        else:
            print("Fuck")
        # Convert the frame to byte format
        frame = buffer.tobytes()
        # Yield the frame in the correct format
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(frame_delay)


# Flask route for the main page
@app.route("/")
def index():
    return render_template("index.html")


# Flask route to generate and stream the video in real time
@app.route("/generate")
def generate():
    #### Extend to have an html load up the audio stream and video stream into a buffer
    #### to play in sync
    input_path = "vid.mp4"  # Update this to the input video path
    return Response(
        load_and_stream_video("./resources/Oneal.jpg", "./resources/BABABOOEY.opus"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    app.run(debug=True)
