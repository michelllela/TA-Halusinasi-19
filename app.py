from flask import Flask, redirect

from compare import bp as compare_bp

app = Flask(__name__)
app.register_blueprint(compare_bp)


@app.route("/")
def home():
    return redirect("/bandingkan")


if __name__ == "__main__":
    app.run(debug=True)