from flask import Flask, jsonify, render_template, render_template_string, request
from jinja2 import TemplateNotFound
from service.github import obter_dados_github 
from dados import dados_personalizados
import pickle
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
usuarios = ["lucianolpsf", "fernandallobao", "jesieldossantos", "Victorrezende19", "calebegomes740", "CaioHarrys", "aucelio0", "brunofluna", "Rafael-ai13", "Xandy77", "pauloalvezz" ] 


@app.route("/")
def home():
    membros = [obter_dados_github(usuario) for usuario in usuarios]
    return render_template("home.html", membros=membros)


@app.route("/<usuario>")
def rota_usuario(usuario):
    return render_template(f"{usuario.lower()}.html")


@app.route("/Victorrezende19/titanic", methods=['GET', 'POST'])
def pred_titanic():
    if request.method == 'POST':
        try:
            pclass = int(request.form['Pclass'])
            age = float(request.form['Age'])
            sibSp = int(request.form['SibSp'])
            parch = int(request.form['Parch'])
            fare = float(request.form['Fare'])
            sex = request.form['Sex']
            embarked = request.form['Embarked']

            sex_encoded = 1 if sex == 'male' else 0  
            embarked_Q = 1 if embarked == 'Q' else 0
            embarked_S = 1 if embarked == 'S' else 0

            features = [[pclass, age, sibSp, parch, fare, sex_encoded, embarked_Q, embarked_S]]

            # Carregar modelo e scaler
            with open('./analises/Victorrezende19/modelo_titanic.pkl', 'rb') as file:
                knn = pickle.load(file)

        

            # Fazer a previsão
            prediction = knn.predict(features)
            resultado = 'Sobreviveu' if prediction == 1 else 'Não Sobreviveu'

            return render_template('Victorrezende19.html', resultado=resultado)

        except Exception as e:
            return f"Ocorreu um erro ao processar a previsão: {e}"

    return render_template('home.html', resultado=None)




@app.route("/lucianolpsf/fruta", methods=['POST'])
def pred_fruta():
    peso = int(request.form['peso'])
    textura =int(request.form['textura'])

    with open('./analises/luciano/modelo_fruta.pkl', 'rb') as file:

        modelo = pickle.load(file)

    fruta =modelo.predict([[peso, textura]])
    return render_template_string(f'sua fruta é: {fruta[0]}')


@app.route("/CalebeGomes740/Predição-de-Diabetes-Calebe", methods=['POST'])
def pred_diabetCa():
    age = int(request.form['age'])
    gender = int(request.form['gender'])
    bmi = float(request.form['bmi']) 
    hypertension = int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    smoking_history_str = request.form['smoking_history']
    smoking_history_map = {
        'never': 0,
        'no_info': 1,
        'current': 2,
        'ever': 3,
        'formerly': 4,
        'not_current': 5
    }
    smoking_history = smoking_history_map.get(smoking_history_str, -1)

    hba1c_level = float(request.form['hba1c_level'])
    blood_glucose_level = float(request.form['blood_glucose_level'])
    

    with open('./analises/calebe/predict_diabets.pkl', 'rb') as file:

        modelo = pickle.load(file)

    features = [[
            age, gender, bmi, hypertension, heart_disease,
            smoking_history, hba1c_level, blood_glucose_level
        ]]
    diabet_predic = modelo.predict(features)
    diabetes_status = diabet_predic[0] # Armazenamos o 0 ou 1 aqui

    # 3. Resposta Aprimorada (renderizando o template HTML)
    if diabetes_status == 1:
        message = "Não há indícios de diabetes."
    else:
        message = "Há indícios de diabetes. Por favor, consulte um médico."

    # Passamos a mensagem e o status para o template HTML
    return render_template('resul_pred_calebe.html', message=message, diabetes_status=diabetes_status)
CORS(app)

# Carregar o dataset
df = pd.read_csv("analises/Caio_Harrys/meu_dataset_completo.csv")
df_filtrado = df.dropna(subset=["plot", "poster"]).reset_index(drop=True)

# Criar matriz TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df_filtrado['plot'])


def recomendar_filmes(titulos_usuario, top_n=9):
    indices_input = []
    for titulo in titulos_usuario:
        match = df_filtrado[df_filtrado['title'].str.lower() == titulo.lower()]
        if match.empty:
            raise ValueError(f"Título '{titulo}' não encontrado no dataset com imagem.")
        indices_input.append(match.index[0])

    vetor_soma = np.asarray(np.sum(tfidf_matrix[indices_input], axis=0))
    similaridades = cosine_similarity(vetor_soma, tfidf_matrix).flatten()
    indices_recomendados = np.argsort(similaridades)[::-1]
    indices_recomendados = [idx for idx in indices_recomendados if idx not in indices_input][:top_n]

    resultados = df_filtrado.loc[indices_recomendados, ['title', 'rating', 'plot', 'poster']]
    return resultados.to_dict(orient='records')


@app.route("/recomendar", methods=["POST"])
def recomendar():
    dados = request.get_json()
    filmes = dados.get("filmes", [])

    try:
        resultados = recomendar_filmes(filmes)
        return jsonify({"recomendacoes": resultados})
    except Exception as e:
        return jsonify({"erro": str(e)}), 400



@app.route("/Rafael-ai13/flores", methods=['POST'])
def pred_flores():
    comprimento_sepala = float(request.form['comprimento_sepala'])
    largura_sepala = float(request.form['largura_sepala'])
    comprimento_petala = float(request.form['comprimento_petala'])	
    largura_petala  = float(request.form['largura_petala'])

    with open('./analises/rafael/predict_flores.pkl', 'rb') as file:

        modelo = pickle.load(file)

    flores =modelo.predict([[comprimento_sepala, largura_sepala, comprimento_petala, largura_petala]])

    # Dicionário para mapear o número para o nome da flor
    nomes_flores = {
        0: 'setosa',
        1: 'versicolor',
        2: 'virginica'
    }

    nome_flor = nomes_flores.get(flores[0], "Flor desconhecida")
    return render_template_string(f'sua flor é: {nome_flor}')

@app.route("/jesieldossantos/analises/jesiel/mcdonalds", methods=['POST'])
def pred_mcdonalds():
    cidade = request.form['cidade']
    bairro = request.form['bairro']
    wifi = int(request.form['wifi'])
    acessibilidade = int(request.form['acessibilidade'])
    delivery = int(request.form['delivery'])
    servicos_com_tempo_extimado = int(request.form['servicos_com_tempo_extimado'])

    # Carregar modelo e LabelEncoders
    caminho_modelo = os.path.join(os.path.dirname(__file__), 'analise', 'Jesiel', 'treinamentomc.pkl')
    with open(caminho_modelo, 'rb') as file:
        modelo = pickle.load(file)
    caminho_le_cidade = os.path.join(os.path.dirname(__file__), 'analise', 'Jesiel', 'le_cidade.pkl')
    with open(caminho_le_cidade, 'rb') as file:
        le_cidade = pickle.load(file)
    caminho_le_bairro = os.path.join(os.path.dirname(__file__), 'analise', 'Jesiel', 'le_bairro.pkl')
    with open(caminho_le_bairro, 'rb') as file:
        le_bairro = pickle.load(file)
    caminho_scaler = os.path.join(os.path.dirname(__file__), 'analise', 'Jesiel', 'scaler.pkl')
    with open(caminho_scaler, 'rb') as file:
        scaler = pickle.load(file)

    # Transformar Cidade e Bairro
    cidade_encoded = le_cidade.transform([cidade])[0]
    bairro_encoded = le_bairro.transform([bairro])[0]

    # Fazer a previsão
    features = [[cidade_encoded, bairro_encoded, wifi, acessibilidade, delivery, servicos_com_tempo_extimado]]
    features_scaled = scaler.transform(features)
    prediction = modelo.predict(features_scaled)
    resultado = 'Oferece serviço' if prediction[0] == 1 else 'Não oferece serviço'

    return render_template('resultado_mcdonalds.html', resultado=resultado)

if __name__ == '__main__':


    app.run(debug=True)