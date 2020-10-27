from flask import render_template, request, json
from demo import app
from demo import ml
import os
import random


model_dir = os.path.join(os.getcwd(), 'demo', 'model')

model_positive = ml.TFModel(model_dir=model_dir,sentiment=True)
model_negative = ml.TFModel(model_dir=model_dir,sentiment=False)

app.config['model_loaded'] = False
app.config['current_model'] = None

@app.route('/', methods=['GET'])
def default():
    return render_template("default.html")


@app.route('/reply', methods=['POST'])
def reply_chat():
    text = request.form['reply']
    if not app.config['model_loaded']:
        if text == "Good":
            app.config['current_model']=model_positive
            predictions= 'So glad to hear that! Can you please share your feedback?'

        elif text== "Bad":
            app.config['current_model']=model_negative
            predictions= 'So sorry to hear that. Could you please share your feedback?'

        app.config['model_loaded']=True
        return json.dumps({'predictions': predictions})

    else:
        # use a random word from the reply text
        tokens = text.split()
        seed_word = random.sample(tokens, k=1)

        # call the TFModel class to predict
        predictions = app.config['current_model'].predict(seed_word)
        print(predictions)
        return json.dumps({'predictions': predictions})
