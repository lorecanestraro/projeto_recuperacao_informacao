from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import string
import os


def carregar_documentos(pasta):
    documentos = []
    nomes = []
    
    for arquivo in os.listdir(pasta):
        if arquivo.endswith(".txt"):
            caminho = os.path.join(pasta, arquivo)
            with open(caminho, "r", encoding="utf-8") as f:
                documentos.append(f.read())
                nomes.append(arquivo)
    
    return documentos, nomes

stopwords_basicas = {
    "o", "a", "e", "de", "do", "da", "para", "no", "na", "pelo", "em",
    "os", "as", "um", "uma", "que", "são"
}


def limpar_texto(frase):
    frase = frase.lower()
    frase = frase.translate(str.maketrans('', '', string.punctuation))
    palavras = frase.split()
    palavras_filtradas = [p for p in palavras if p not in stopwords_basicas]
    return " ".join(palavras_filtradas)


def main():
    pasta_docs = "docs"
    
    documentos, nomes_arquivos = carregar_documentos(pasta_docs)
    documentos_proc = [limpar_texto(doc) for doc in documentos]

    print("Documentos pré-processados:\n")
    for i, doc in enumerate(documentos_proc, start=1):
        print(f"{nomes_arquivos[i-1]}: {doc}")
    print("\n" + "="*60 + "\n")


    count_vectorizer = CountVectorizer()
    matriz_tf = count_vectorizer.fit_transform(documentos_proc)
    vocabulario = count_vectorizer.get_feature_names_out()

    df_tf = pd.DataFrame(
        matriz_tf.toarray().T,
        index=vocabulario,
        columns=nomes_arquivos
    )

    print("Matriz Termo-Documento (TF):\n")
    print(df_tf)
    print("\n" + "="*60 + "\n")

    tfidf_vectorizer = TfidfVectorizer(norm='l2')
    matriz_tfidf = tfidf_vectorizer.fit_transform(documentos_proc)
    vocabulario_tfidf = tfidf_vectorizer.get_feature_names_out()

    df_tfidf = pd.DataFrame(
        matriz_tfidf.toarray().T,
        index=vocabulario_tfidf,
        columns=nomes_arquivos
    )

    print("Matriz TF-IDF:\n")
    print(df_tfidf.round(4))
    print("\n" + "="*60 + "\n")


    consulta = input("🔍 Digite sua consulta: ")
    consulta_proc = limpar_texto(consulta)

    print("\nConsulta processada:", consulta_proc)
    print("\n" + "="*60 + "\n")

    vetor_consulta = tfidf_vectorizer.transform([consulta_proc])

    df_consulta = pd.DataFrame(
        vetor_consulta.toarray(),
        columns=vocabulario_tfidf,
        index=["Consulta"]
    )

    print("Vetor TF-IDF da consulta:\n")
    print(df_consulta.round(4))
    print("\n" + "="*60 + "\n")


    similaridades = cosine_similarity(vetor_consulta, matriz_tfidf)[0]

    ranking = sorted(
        [(i, score) for i, score in enumerate(similaridades)],
        key=lambda x: x[1],
        reverse=True
    )

    print("Ranking dos documentos:\n")
    for pos, score in ranking:
        print(f"{nomes_arquivos[pos]} - Score: {score:.4f}")
        print(documentos[pos])
        print("-" * 50)

if __name__ == "__main__":
    main()