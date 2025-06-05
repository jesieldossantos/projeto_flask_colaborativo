from flask import Flask, render_template, render_template_string, request
from jinja2 import TemplateNotFound
from service.github import obter_dados_github 
from dados import dados_personalizados
import pickle

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


if __name__ == '__main__':
    app.run(debug=True)