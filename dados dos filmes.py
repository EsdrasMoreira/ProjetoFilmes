import pandas as pd
from caminho import caminho
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from scipy.stats import f_oneway, f
import numpy as np

movies = pd.read_csv(f'{caminho}\\tmdb_5000_movies.csv') #carregando os dados
credits = pd.read_csv(f"{caminho}\\tmdb_5000_credits.csv")#carregando os dados


df = movies.merge(credits, left_on='id', right_on='movie_id', how='left')#Juntando as duas tabelas em um so DataFrame

df['lucro'] = df['revenue'] - df['budget'] #cálculo do lucro e acrescentando na tabela

df = df.rename(columns={ #renomeando os titulos 
    'title_x' : 'titulo_filme',
    'budget' : 'orçamento',
    'revenue' : 'receita',
    'lucro' : 'lucro',
    'vote_average' : 'nota_media',
    'vote_count' : 'numero_de_votos'
})

#novo dataframe sobre os lucros
df_lucro = df[['titulo_filme', 'orçamento', 'receita', 'lucro']].sort_values(by='lucro', ascending=False)

#os 10 filmes mais lucrativos
print (df_lucro.head(10))

#limpando os dados (eliminando os dados de orçamento e receita faltantes) 
df = df[(df['orçamento'] > 0) & (df['receita'] > 0)]

#novo dataframe sobre os prejuízos
df_prejuizo = df[['titulo_filme', 'orçamento', 'receita', 'lucro']].sort_values(by='lucro', ascending=True)

#os 10 filmes menos lucrativos
top10 = df.sort_values(by='lucro', ascending=False).head(10)

plt.bar(top10['titulo_filme'], top10['lucro'], color='green', width=0.6)

plt.xticks(top10['titulo_filme'], rotation=25, ha='right')

plt.xlabel('Filme')

plt.ylabel('Lucro (Em milhões de Dólares)')

plt.title('Os 10 filmes mais lucrativos da história')

plt.tight_layout()

plt.show()

#montando o gráfico dos menos lucrativos

preju10 = df.sort_values(by='lucro', ascending=True).head(10)

plt.figure(figsize=(12,6))

plt.bar(preju10['titulo_filme'], preju10['lucro'], color='red',width=0.6)

plt.xticks(preju10['titulo_filme'], rotation=25, ha='right')

plt.xlabel('Filme')

plt.ylabel('Prejuízo (Em milhões de Dólares)')

plt.title('Os 10 filmes menos lucrativos da história')

plt.tight_layout()

plt.show()

#Filmes com maior orçamento, tendem a ter maior nota?

df_OrcNota = df[['orçamento', 'nota_media', 'numero_de_votos']] #criando um dataframe com apenas as duas variáveis

df_OrcNota = df_OrcNota.dropna() #limpando as linhas vazias

df_OrcNota = df_OrcNota[(df_OrcNota['orçamento'] > 0) & #Filtrando os dados para evitar distorção
                        (df_OrcNota['numero_de_votos'] > 100) &
                         (df_OrcNota['nota_media'] > 0)
]

#coeficiente pearson
correlacao = df_OrcNota['orçamento'].corr(df_OrcNota['nota_media'])
print(f'Coeficiente de Correlação de Pearson (Orçamento x Nota_Media): {correlacao:.4f}')

#plotando o gráfico de dispersão com regressão linear
plt.figure(figsize=(12,6))

sns.regplot(data=df_OrcNota, x='orçamento', y='nota_media', scatter_kws={'alpha' : 0.5}, line_kws={'color':'red'})

plt.title('Correlação entre Orçamento e Nota Média dos filmes')
plt.xlabel('Orçamento (Em Dólares)')
plt.ylabel('Nota dos Filmes')
plt.grid(True)

plt.show()

#Conclusão, Há uma correlação negativa muito fraca, ou seja, quase insignificante.
#Não tem influência entre as duas variáveis, ambos são dados independentes.
#O Sucesso na crítica não está ligeiramente ligado à quantidade investida.

# O gênero do filme, influencia na nota?

df['genres'] = df['genres'].apply(ast.literal_eval) #Transformando a string para uma lista de verdade

#Criando uma coluna com o gênero principal de cada filme
df['genero_principal'] = df['genres'].apply(lambda x: x[0]['name'] if len(x) > 0 else 'Desconhecido')

traducoes_genero = {
    'Action': 'Ação',
    'Adventure': 'Aventura',
    'Animation': 'Animação',
    'Comedy': 'Comédia',
    'Crime': 'Crime',
    'Documentary': 'Documentário',
    'Drama': 'Drama',
    'Family': 'Família',
    'Fantasy': 'Fantasia',
    'Foreign': 'Estrangeiro',
    'History': 'História',
    'Horror': 'Terror',
    'Music': 'Musical',
    'Mystery': 'Mistério',
    'Romance': 'Romance',
    'Science Fiction': 'Ficção Científica',
    'TV Movie': 'Filme de TV',
    'Thriller': 'Suspense',
    'War': 'Guerra',
    'Western': 'Faroeste',
    'Desconhecido': 'Desconhecido'
}

df['genero_principal'] = df['genero_principal'].map(traducoes_genero) 

#Agrupando os elementos pelo gênero principal e ordenando a média deles de forma decrescente
media_genero = df.groupby('genero_principal')['nota_media'].mean().sort_values(ascending=False)

# plotando o grafico para visualizar as notas por gênero

plt.figure(figsize=(12,6))
media_genero.plot(kind= 'barh', color='skyblue')
plt.xlabel('Nota Média')
plt.ylabel('Gênero')
plt.title('Nota média dos filmes por gênero')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.5)
plt.show()

#Conclusão: Analisando o gráfico, ele sugere que pode haver sim uma relação entre gêneros e sucesso de crítica.
#Porém, não há como afirmar com certeza, apenas com esse grafico. É necessário outros testes estatísticos
#para se tirar uma conclusão

grupos = df.groupby('genero_principal')['nota_media'].apply(list)

f_stats, p_value = f_oneway(*grupos)

significancia = 0.05

k = df['genero_principal'].nunique()
n = df['nota_media'].count()
gl1 = k - 1 #graus de liberade do numerador
gl2 = n - k #graus de liberdade do denominador

valor_critico = f.ppf(1 - significancia, gl1,gl2)#calculo do valor critico


print(f'Estatistica F: {f_stats:.4f}')
print(f'p valor: {p_value:.4f}')
print(f'Valor crítico de F, com {significancia} de significância: {valor_critico:.4f}')
#HA: Há evidências estatísticas fortes de que pelo menos um gênero tem uma nota média significativamente diferente dos outros.
#H0: As médias das notas médias dos filmes são iguais entre todos os gêneros.
#Conclusão: como o valor p < alfa, rejeitamos a hipótese nula

#A língua original do filme, influencia em sua nota?

traducao_idiomas = {
    'pt': 'Português',
    'he': 'Hebraico',
    'fa': 'Persa',
    'te': 'Telugu',
    'id': 'Indonésio',
    'it': 'Italiano',
    'ro': 'Romeno',
    'de': 'Alemão',
    'ja': 'Japonês',
    'xx': 'Desconhecido/Outro',
    'pl': 'Polonês',
    'nl': 'Holandês',
    'hi': 'Hindi',
    'es': 'Espanhol',
    'is': 'Islandês',
    'af': 'Africâner',
    'da': 'Dinamarquês',
    'fr': 'Francês',
    'nb': 'Norueguês Bokmål',
    'cn': 'Chinês (geral)',
    'th': 'Tailandês',
    'ko': 'Coreano',
    'zh': 'Chinês',
    'en': 'Inglês',
    'no': 'Norueguês',
    'ru': 'Russo',
    'vi': 'Vietnamita'
}

df['original_language'] = df['original_language'].map(traducao_idiomas)

media_idioma = df.groupby('original_language')['nota_media'].mean().sort_values(ascending=False)

quantidade_idioma = df['original_language'].value_counts()

#plotando o gráfico (idiomas x notas)

plt.figure(figsize=(12,6))
media_idioma.plot(kind='barh', color='skyblue')
plt.xlabel('Nota Média')
plt.ylabel('Idioma Original')
plt.title('Nota Média por Idioma Original')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.show()

print(quantidade_idioma)

#Conclusão: Embora pareça que filmes de lingua "estrangeira", isto é, lingua diferente do inglês, tenha notas superiores
#aos filmes de língua inglesa, enquanto há apenas 2 filmes de lingua portuguesa (maior média), no dataset,
#há 3102 filmes de lingua inglesa. Essa desproporcionalidade pode comprometer a validade estatística das comparações,
#devido ao número discrepante das amostras entre os grupos. Sendo assim, não é possível concluir com segurança
#que o idioma original realmente influencie no sucesso das críticas.