import pickle
from flask import Flask, jsonify, request

species = ['Adelie', 'Chinstrap', 'Gentoo']

def preprocess_penguin(penguin):
    island_categories = ['Biscoe', 'Dream', 'Torgersen']
    sex_categories = ['MALE', 'FEMALE']
    
    island_encoded = [1 if penguin['island'] == category else 0 for category in island_categories]
    sex_encoded = [1 if penguin['sex'] == category else 0 for category in sex_categories]

    # Return the numeric features with the one-hot encoded columns
    return [
            penguin["body_mass_g"],
            penguin["culmen_depth_mm"],
            penguin["culmen_length_mm"],
            penguin["flipper_length_mm"],
        ] + island_encoded + sex_encoded

def predict_single(penguin, sc, model):
    penguin_features = preprocess_penguin(penguin)
    penguin_std = sc.transform([penguin_features])
    
    y_pred = model.predict(penguin_std)[0]
    y_prob = model.predict_proba(penguin_std)[0][y_pred]

    return (y_pred, y_prob)

def predict(sc, model):
    penguin = request.get_json()
    species_index, probability = predict_single(penguin, sc, model)
   
    result = {
        'Especie': species[species_index],
        'Probabilidad': float(probability)
    }

    return jsonify(result)

app = Flask('penguins')


@app.route('/predict_lr', methods=['POST'])
def predict_lr():
    with open('models/lr.pck', 'rb') as f:
        sc, model = pickle.load(f)
    return predict(sc,model)

@app.route('/predict_svm', methods=['POST'])
def predict_svm():
    with open('models/svm.pck', 'rb') as f:
        sc, model = pickle.load(f)
    return predict(sc,model)

@app.route('/predict_dt', methods=['POST'])
def predict_dt():
    with open('models/dt.pck', 'rb') as f:
        sc, model = pickle.load(f)
    return predict(sc,model)

@app.route('/predict_knn', methods=['POST'])
def predict_knn():
    with open('models/knn.pck', 'rb') as f:
        sc, model = pickle.load(f)
    return predict(sc,model)


if __name__ == '__main__':
    app.run(debug=True, port=8000)