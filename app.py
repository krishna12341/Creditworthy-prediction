import numpy as np
from flask import Flask, request, render_template
import pickle
app = Flask(__name__,template_folder='Template')

random_forest = pickle.load(open('model2.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')
#prediction function
def ValuePredictor(to_predict_list):
    to_predict=np.array(to_predict_list).reshape(1,20)
    random_forest = pickle.load(open('model2.pkl', 'rb'))
    result = random_forest.predict(to_predict)
    return result[0]
@app.route('/result', methods =['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)
        if int(result) == 1:
            prediction = 'Eligible to taking credit loan'
        else:
            prediction ='Not eligible to taking credit loan'
        return render_template('result.html',prediction = prediction)












print(__name__)
if __name__ == "__main__":
    app.run(debug=True,port=8001)