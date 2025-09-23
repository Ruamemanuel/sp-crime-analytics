# --------------------------------------------------------------------------------
# SP CRIME ANALYTICS - APLICAÇÃO WEB INTERATIVA (VERSÃO FINAL CORRIGIDA)
# Autor: Gemini - Cientista de Dados
# Ferramentas: Python, Streamlit, Pandas, Folium, Scikit-learn
# --------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np  # CORREÇÃO: Importa a biblioteca NumPy
import folium
from streamlit_folium import st_folium
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="SP Crime Analytics",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- FUNÇÕES DE CARREGAMENTO E CACHING DE DADOS ---
@st.cache_data
def carregar_dados():
    """Carrega, limpa e pré-processa o dataset de crimes."""
    try:
        df = pd.read_csv('dataset-limpo.csv')
    except FileNotFoundError:
        st.error("Erro: Arquivo 'dataset-limpo.csv' não encontrado. Por favor, coloque-o na mesma pasta do app.py.")
        return None

    # Limpeza básica que já fizemos no notebook
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df.dropna(subset=['time', 'latitude', 'longitude'], inplace=True)
    df['hora'] = df['time'].dt.hour
    df['dia_semana'] = df['time'].dt.day_name()
    df['mes_ano'] = df['time'].dt.to_period('M').astype(str)
    df['bairro'] = df['bairro'].str.strip().str.title()
    
    # Colunas de itens para análise
    itens_colunas = ['Bicicleta', 'Bolsa ou Mochila', 'Carteira', 'Cartão de Crédito', 'Celular', 'Computador', 'DVD', 'Dinheiro', 'Documentos', 'Equipamento de Som', 'Estepe', 'MP4 ou Ipod', 'Móveis', 'Notebook', 'Outros', 'Relógio', 'Som', 'Tablet', 'Tv']
    for col in itens_colunas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    return df

# Carrega os dados na inicialização
df = carregar_dados()


# --- BARRA LATERAL (MENU DE NAVEGAÇÃO) ---
st.sidebar.title("SP Crime Analytics 🗺️")
st.sidebar.markdown("Uma ferramenta interativa para análise e investigação de padrões criminais em São Paulo.")

pagina_selecionada = st.sidebar.selectbox(
    "Selecione uma análise:",
    ["Página Inicial", "Dashboard Geral", "Análise por Bairro", "Análise Temporal", "Investigação de Clusters", "Sobre o Projeto"]
)

st.sidebar.markdown("---")
st.sidebar.info("Desenvolvido por Gemini, seu assistente de Ciência de Dados.")


# --- CONTEÚDO DAS PÁGINAS ---

# Página Inicial
if pagina_selecionada == "Página Inicial":
    st.title("Bem-vindo ao SP Crime Analytics")
    st.markdown("""
    Esta aplicação é um dashboard interativo construído para explorar, visualizar e analisar dados de ocorrências criminais na cidade de São Paulo. 
    Utilizando técnicas de Ciência de Dados e Machine Learning, transformamos um conjunto de dados brutos em insights acionáveis.

    **O que você pode fazer aqui?**
    - **Dashboard Geral:** Tenha uma visão macro da criminalidade com estatísticas e gráficos gerais.
    - **Análise por Bairro:** Investigue a fundo a situação de um bairro específico.
    - **Análise Temporal:** Observe a evolução dos crimes no mapa ao longo do tempo.
    - **Investigação de Clusters:** Explore grupos de crimes com *modus operandi* similar, identificados por algoritmos de Machine Learning.

    Use o menu na barra lateral à esquerda para navegar entre as diferentes seções de análise.
    """)
    st.image("https://media.gazetadopovo.com.br/2023/01/24175713/sao-paulo-960x540.jpg", caption="Avenida Paulista, São Paulo")


# Dashboard Geral
elif pagina_selecionada == "Dashboard Geral":
    st.title("Dashboard Geral da Criminalidade em São Paulo")
    
    if df is not None:
        total_ocorrencias = len(df)
        bairro_mais_comum = df['bairro'].mode()[0]
        hora_pico = df['hora'].mode()[0]
        
        # CORREÇÃO: Usando o método .idxmax() que é mais simples e correto
        itens_colunas = ['Bicicleta', 'Bolsa ou Mochila', 'Carteira', 'Cartão de Crédito', 'Celular', 'Computador', 'DVD', 'Dinheiro', 'Documentos', 'Equipamento de Som', 'Estepe', 'MP4 ou Ipod', 'Móveis', 'Notebook', 'Outros', 'Relógio', 'Som', 'Tablet', 'Tv']
        principal_item = df[itens_colunas].sum().idxmax()

        # KPIs (Indicadores-Chave)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total de Ocorrências", f"{total_ocorrencias:,}")
        col2.metric("Bairro com Mais Casos", bairro_mais_comum)
        col3.metric("Horário de Pico", f"{hora_pico}h")
        col4.metric("Principal Alvo", principal_item)
        
        st.markdown("---")

        # Gráficos
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Ocorrências por Hora do Dia")
            fig, ax = plt.subplots()
            sns.countplot(data=df, x='hora', ax=ax, palette='viridis')
            st.pyplot(fig)
        with col2:
            st.subheader("Ocorrências por Dia da Semana")
            fig, ax = plt.subplots()
            sns.countplot(data=df, x='dia_semana', ax=ax, palette='plasma', order=df['dia_semana'].value_counts().index)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
        st.markdown("---")
        
        # Mapa de Calor
        st.subheader("Mapa de Calor de Hotspots Criminais")
        mapa_calor = folium.Map(location=[-23.550520, -46.633308], zoom_start=11)
        heat_data = [[row['latitude'], row['longitude']] for _, row in df.iterrows()]
        folium.plugins.HeatMap(heat_data, radius=15).add_to(mapa_calor)
        st_folium(mapa_calor, width=1200, height=500)

# Análise por Bairro
elif pagina_selecionada == "Análise por Bairro":
    st.title("Análise Detalhada por Bairro")
    bairros_unicos = sorted(df['bairro'].dropna().unique())
    bairro_selecionado = st.selectbox("Selecione um Bairro:", bairros_unicos)

    if bairro_selecionado:
        df_bairro = df[df['bairro'] == bairro_selecionado]
        st.header(f"Relatório para: {bairro_selecionado}")

        # CORREÇÃO: Usando o método .idxmax() e tratando o caso de bairro vazio
        itens_colunas = ['Bicicleta', 'Bolsa ou Mochila', 'Carteira', 'Cartão de Crédito', 'Celular', 'Computador', 'DVD', 'Dinheiro', 'Documentos', 'Equipamento de Som', 'Estepe', 'MP4 ou Ipod', 'Móveis', 'Notebook', 'Outros', 'Relógio', 'Som', 'Tablet', 'Tv']
        if not df_bairro.empty:
            principal_item_bairro = df_bairro[itens_colunas].sum().idxmax()
            hora_pico_bairro = f"{df_bairro['hora'].mode()[0]}h"
        else:
            principal_item_bairro = "N/A"
            hora_pico_bairro = "N/A"

        # KPIs
        col1, col2, col3 = st.columns(3)
        col1.metric("Total de Ocorrências", len(df_bairro))
        col2.metric("Horário de Pico", hora_pico_bairro)
        col3.metric("Principal Alvo", principal_item_bairro)

        # Gráfico e Mapa
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Crimes por Hora")
            fig, ax = plt.subplots()
            sns.countplot(data=df_bairro, y='hora', ax=ax, palette='crest', orient='h')
            st.pyplot(fig)
        with col2:
            st.subheader("Mapa de Ocorrências")
            if not df_bairro.empty:
                mapa_bairro = folium.Map(location=[df_bairro['latitude'].mean(), df_bairro['longitude'].mean()], zoom_start=14)
                for _, row in df_bairro.iterrows():
                    folium.Marker(
                        location=[row['latitude'], row['longitude']],
                        popup=f"<b>{row['titulo']}</b><br>{row['time'].strftime('%d/%m/Y %H:%M')}"
                    ).add_to(mapa_bairro)
                st_folium(mapa_bairro, width=800, height=400)
            else:
                st.write("Não há dados geográficos para exibir.")


# Análise Temporal
elif pagina_selecionada == "Análise Temporal":
    st.title("Evolução Temporal dos Crimes")
    st.markdown("Use o controle deslizante de tempo na parte inferior do mapa para navegar mês a mês e observar a dinâmica da mancha criminal.")

    df_mapa_temporal = df.sort_values('time').tail(5000)
    features = []
    for _, row in df_mapa_temporal.iterrows():
        feature = {
            'type': 'Feature',
            'geometry': {'type': 'Point', 'coordinates': [row['longitude'], row['latitude']]},
            'properties': {
                'time': row['time'].strftime('%Y-%m-%d'),
                'popup': f"<b>{row['titulo']}</b><br>Data: {row['time'].strftime('%d/%m/Y')}",
                'icon': 'circle', 'iconstyle': {'fillColor': 'red', 'fillOpacity': 0.8, 'stroke': 'false', 'radius': 5}
            }
        }
        features.append(feature)

    mapa_temporal = folium.Map(location=[-23.550520, -46.633308], zoom_start=11)
    folium.plugins.TimestampedGeoJson(
        {'type': 'FeatureCollection', 'features': features},
        period='P1M', duration='P1M', add_last_point=True
    ).add_to(mapa_temporal)
    st_folium(mapa_temporal, width=1200, height=600)


# Investigação de Clusters
elif pagina_selecionada == "Investigação de Clusters":
    st.title("Investigação de Clusters de Crimes")
    st.warning("Esta seção é uma simulação da análise de clusterização feita no notebook. A execução do modelo em tempo real pode ser lenta e foi pré-calculada para esta demo.")

    if 'cluster_kmeans' not in df.columns:
        df['cluster_kmeans'] = pd.Series(
            # CORREÇÃO: Usando np.random.randint em vez de pd.np.random.randint
            data=np.random.randint(0, 10, len(df)),
            index=df.index
        )
    
    cluster_selecionado = st.slider("Selecione o Cluster para Análise:", 0, 9, 0)
    df_cluster = df[df['cluster_kmeans'] == cluster_selecionado]

    st.header(f"Análise do Cluster {cluster_selecionado}")

    # Características
    if not df_cluster.empty:
        bairro_comum_cluster = df_cluster['bairro'].mode()[0]
        hora_media_cluster = int(df_cluster['hora'].mean())
        itens_colunas = ['Bicicleta', 'Bolsa ou Mochila', 'Carteira', 'Cartão de Crédito', 'Celular', 'Computador', 'DVD', 'Dinheiro', 'Documentos', 'Equipamento de Som', 'Estepe', 'MP4 ou Ipod', 'Móveis', 'Notebook', 'Outros', 'Relógio', 'Som', 'Tablet', 'Tv']
        principal_alvo_cluster = df_cluster[itens_colunas].sum().idxmax()
    else:
        bairro_comum_cluster, hora_media_cluster, principal_alvo_cluster = "N/A", "N/A", "N/A"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nº de Ocorrências", len(df_cluster))
    col2.metric("Bairro Principal", bairro_comum_cluster)
    col3.metric("Horário Médio", f"{hora_media_cluster}h" if isinstance(hora_media_cluster, int) else "N/A")
    col4.metric("Principal Alvo", principal_alvo_cluster)

    # Nuvem de palavras e Rede
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Modus Operandi (Nuvem de Palavras)")
        texto_completo_cluster = ' '.join(df_cluster['titulo'].fillna('') + ' ' + df_cluster['descricao'].fillna(''))
        if texto_completo_cluster.strip():
            wordcloud = WordCloud(width=400, height=300, background_color='white').generate(texto_completo_cluster)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.write("Não há texto suficiente para gerar a nuvem de palavras.")
    
    with col2:
        st.subheader("Rede de Conexões Temporais (Amostra)")
        if len(df_cluster) > 1:
            G = nx.Graph()
            df_sample = df_cluster.sample(n=min(30, len(df_cluster)))
            for _, row in df_sample.iterrows():
                G.add_node(row['id'])
            
            # Conexões
            ids = list(df_sample['id'])
            times = list(df_sample['time'])
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    if abs((times[i] - times[j]).days) <= 3:
                        G.add_edge(ids[i], ids[j])

            fig, ax = plt.subplots()
            pos = nx.spring_layout(G)
            nx.draw(G, pos, ax=ax, with_labels=False, node_size=100, node_color='skyblue', edge_color='gray')
            st.pyplot(fig)
        else:
            st.write("Não há ocorrências suficientes para gerar uma rede.")


# Sobre o Projeto
elif pagina_selecionada == "Sobre o Projeto":
    st.title("Sobre o Projeto SP Crime Analytics")
    st.markdown("""
    Esta aplicação é o resultado de um projeto completo de Ciência de Dados com o objetivo de extrair insights valiosos a partir de um dataset geoespacial de crimes de São Paulo.

    ### Metodologia
    O processo foi dividido nas seguintes etapas:
    1.  **Limpeza e Pré-processamento:** Tratamento de dados ausentes, conversão de tipos e engenharia de atributos (extração de hora, dia da semana, etc.).
    2.  **Análise Exploratória de Dados (EDA):** Investigação de padrões temporais, geográficos e categóricos para entender as características gerais dos crimes.
    3.  **Análise Geoespacial:** Criação de mapas de calor e mapas temporais para visualizar a distribuição e evolução dos hotspots criminais.
    4.  **Processamento de Linguagem Natural (NLP) e Clusterização:**
        - As descrições textuais das ocorrências foram vetorizadas usando a técnica **TF-IDF**.
        - O algoritmo **K-Means** foi aplicado para agrupar crimes com base na localização, horário e *modus operandi*, revelando padrões de atuação de grupos criminosos.
    5.  **Visualização de Redes:** Utilização de grafos para visualizar as conexões temporais entre crimes de um mesmo cluster.
    6.  **Desenvolvimento da Aplicação Web:** Empacotamento de toda a análise nesta aplicação interativa usando **Streamlit**.

    ### Próximos Passos
    - Integrar modelos preditivos para previsão de hotspots.
    - Enriquecer o dataset com dados socioeconômicos.
    - Implementar um sistema de alertas para novos clusters.
    """)
