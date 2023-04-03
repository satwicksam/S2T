from flask import *
# import os
# from werkzeug.utils import secure_filename
import whisper
import librosa

# dir = os.path.dirname(os.path.abspath(__file__))
# uploads_dir = os.path.join(dir, 'uploads')
# os.makedirs(uploads_dir, exists_ok=True)

app=Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def home():
    return render_template("index.html")

@app.route("/speech-to-text", methods=["POST", "GET"])
def main():
    if request.method=="POST":
        if request.form["run"]=="Run Program":
            f=request.files["upfile"]
            f.save("uploads/"+f.filename)
            # f.save(os.path.join(uploads_dir, secure_filename(f.filename)))
            # audio_src = os.path.join(uploads_dir, secure_filename(f.filename))
            audio_src = ("uploads/"+f.filename)
            model = whisper.load_model("base")
            audio, sr = librosa.load(audio_src)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(model.device) 
            _, probs = model.detect_language(mel)
            lang=f"Detected Language : {max(probs, key=probs.get)}"
            options = whisper.DecodingOptions(fp16 = False)
            result = whisper.decode(model, mel, options)
            text=result.text
            return render_template("index.html", language=lang, text=text, audio=f.filename)

if __name__=="__main__":
    app.run()