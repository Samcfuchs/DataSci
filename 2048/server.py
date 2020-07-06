from flask import Flask
from flask import render_template, url_for, jsonify

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('index.html')
