from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello"

@app.route('/get-prediction', methods=['POST'])
def get_prediction():
    data = request.json

    if 'expences' in data:
        try:
            model = joblib.load('expense_assessment_model.pkl')
            try:

                predTemp = []
                input_data = data['expences']
                for item in input_data:
                    pred = model.predict([item])
                    predTemp.append(
                        {
                            'category-id' : item[0],
                            'prediction' : pred[0]
                        }
                    )
                return jsonify({predTemp})
            except Exception as e:
                print("Error predicting:", e)
        except Exception as e:
            print("Error loading the model:", e)
            exit(1)




if __name__ == '__main__':
   app.run()