from flask import Flask, jsonify, render_template, render_template_string, request
from jinja2 import TemplateNotFound
from service.github import obter_dados_github 
from dados import dados_personalizados
import pickle
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
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
    dados = dados_personalizados.get(usuario, {"titulo": f"Perfil de {usuario}", "conteudo": "Nenhum dado específico disponível."})
    try:
        return render_template(f"{usuario}.html",  dados=dados)
    except TemplateNotFound:
        return render_template_string(f"<h1>{dados['titulo']}</h1><p>{dados['conteudo']}</p>") 


@app.route("/lucianolpsf/fruta", methods=['POST'])
def pred_fruta():
    peso = int(request.form['peso'])
    textura =int(request.form['textura'])

    with open('./analises/luciano/modelo_fruta.pkl', 'rb') as file:

        modelo = pickle.load(file)

    fruta =modelo.predict([[peso, textura]])
    return render_template_string(f'sua fruta é: {fruta[0]}')


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




if __name__ == '__main__':
    app.run(debug=True)