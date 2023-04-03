from flask import *
import os
from werkzeug.utils import secure_filename
import whisper
import librosa

dir = os.path.dirname(os.path.abspath(__file__))
uploads_dir = os.path.join(dir, 'uploads')
# os.makedirs(uploads_dir, exists_ok=True)

app=Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def home():
    return render_template("""
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="/static/style.css">
    <title>Speech to Text Converter</title>
</head>

<body>
    <div class="main_page">
        <div class="header_text">Speech To Text Converter</div>
        <form action="/speech-to-text" method="post" enctype="multipart/form-data">
            <input type="file" name="upfile" required accept=".wav, .mp3" class="btn1"><br>
            <input type="submit" value="Run Program" name="run" class="btn"><br>
        </form>
        <div>
            <div class="dec_lan">{{ language }}</div>
            <h3>Detected Text:</h3> 
            <textarea id="out_text" placeholder="Output will show here..">{{ text }}</textarea><br>
            <div>
                <button onclick="copyText()" class="btn">Copy Text</button><br>
                <button onclick="downloadCSV()" class="btn">Download .CSV</button><br>
            </div>
        </div>
    </div>



    <script>
        function copyText() {
            var copyText = document.getElementById("out_text");
            copyText.select();
            navigator.clipboard.writeText(copyText.value);
            alert("Copied...!");
        }

        function downloadCSV(){
            var a = document.createElement('a');
            with (a) {
                href='data:text/csv;base64,' + btoa(document.getElementById('out_text').value);
                download='Output.csv';
            }
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        }
    </script>
</body>

</html>
                           """)

@app.route("/speech-to-text", methods=["POST", "GET"])
def main():
    if request.method=="POST":
        if request.form["run"]=="Run Program":
            f=request.files["upfile"]
            # f.save(f.filename)
            f.save(os.path.join(uploads_dir, secure_filename(f.filename)))
            audio_src = os.path.join(uploads_dir, secure_filename(f.filename))
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