Minimal Vector Space Model (TF-IDF & Cosine Similarity)

Este projeto implementa um Modelo de Espaço Vetorial básico para Recuperação de Informação (IR).
Ele processa um diretório de arquivos de texto, constrói matrizes de frequência de termos (TF) e ponderação TF-IDF,
e permite realizar consultas textuais interativas ordenadas por Similaridade de Cosseno.

🚀 Funcionalidades

Pipeline de Pré-processamento: Limpeza de strings (conversão para minúsculas, remoção de pontuação) e filtragem de stopwords customizadas em português.
Vetorização Dupla:
Geração de Matriz Termo-Documento bruta (Frequência de Termos - TF).
Geração de Matriz ponderada por relevância estatística (TF-IDF com normalização $L_2$).
Mecanismo de Busca: Transforma consultas do usuário no mesmo espaço vetorial dos documentos e calcula o score de relevância via similaridade de cosseno.
Exibição Transparente: Exibe no terminal os DataFrames do Pandas contendo o vocabulário e os pesos calculados para inspeção direta.



