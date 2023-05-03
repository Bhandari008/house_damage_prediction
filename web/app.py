from flask import Flask, render_template, request

from web.utils.feature_importance import save_importance, feature_names, make_predictions



app = Flask(__name__)





# home page
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/feat_imp', methods = ['GET', 'POST'])
def feature_importance():
    if request.method == 'POST':
        algorithm_name = request.form['algorithm']
        # make a function call to plot image
        if algorithm_name == 'softmax':
            image_name = save_importance(algorithm_name)
        elif algorithm_name == 'decisiontree':
            image_name = save_importance(algorithm_name)
        else:
            render_template('index.html')

    return render_template('feature_imp.html', image = image_name)



@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('input_form.html', feature_names=feature_names)
    if request.method == 'POST':
        algorithm_name = request.form.get("algorithm")
        feature_values = [request.form[feature_name] for feature_name in feature_names]
        predicted_grade = make_predictions(algorithm_name, feature_values)
        if predicted_grade[0] == 3:
            damage_grade = 'Your house or building may suffer almost complete destruction'
            color = 'red'
        elif predicted_grade[0] == 2:
            damage_grade = 'Your house or building may suffer medium amount of damage'
            color = 'green'
        else:
            damage_grade = 'Your house or building may suffer low damage'
            color = 'grey'
        #epresents low damage
        #2 represents a medium amount of damage
        # 3 represents almost complete destruction
        return render_template('index.html', prediction = damage_grade, color = color)































if __name__ == '__main___':
    app.run()
