# --------------------------------------------------------------------------------
# SP CRIME ANALYTICS - APLICAÇÃO WEB PROFISSIONAL (VERSÃO 2.1 - CORRIGIDA)
# Autor: Gemini - Cientista de Dados Sênior
# Descrição: Dashboard robusto para análise interativa de dados criminais,
#            focado em clareza, insights e usabilidade para todos os públicos.
# --------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="SP Crime Analytics | Dashboard Profissional",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FUNÇÕES DE CARREGAMENTO E PRÉ-PROCESSAMENTO ---
@st.cache_data
def carregar_dados():
    """
    Carrega, limpa e pré-processa o dataset de crimes.
    A função retorna um DataFrame do Pandas pronto para análise.
    """
    try:
        df = pd.read_csv('dataset-limpo.csv')
    except FileNotFoundError:
        st.error("Erro Crítico: Arquivo 'dataset-limpo.csv' não encontrado. Por favor, coloque-o na mesma pasta do app.py.")
        return None

    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df.dropna(subset=['time', 'latitude', 'longitude'], inplace=True)
    df['hora'] = df['time'].dt.hour
    df['dia_semana'] = df['time'].dt.day_name()
    df['mes_ano'] = df['time'].dt.to_period('M').astype(str)
    df['bairro'] = df['bairro'].str.strip().str.title()
    
    itens_para_processar = ['Bicicleta', 'Bolsa ou Mochila', 'Carteira', 'Cartão de Crédito', 'Celular', 'Computador', 'DVD', 'Dinheiro', 'Documentos', 'Equipamento de Som', 'Estepe', 'MP4 ou Ipod', 'Móveis', 'Notebook', 'Outros', 'Relógio', 'Som', 'Tablet', 'Tv']
    for col in itens_para_processar:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    return df

# Executa o carregamento dos dados
df = carregar_dados()

# --- VARIÁVEL GLOBAL PARA COLUNAS DE ITENS ---
# CORREÇÃO: Definindo a lista de colunas aqui para que fique acessível a todo o script.
itens_colunas = ['Bicicleta', 'Bolsa ou Mochila', 'Carteira', 'Cartão de Crédito', 'Celular', 'Computador', 'DVD', 'Dinheiro', 'Documentos', 'Equipamento de Som', 'Estepe', 'MP4 ou Ipod', 'Móveis', 'Notebook', 'Outros', 'Relógio', 'Som', 'Tablet', 'Tv']


# --- BARRA LATERAL (MENU DE NAVEGAÇÃO) ---
st.sidebar.title("SP Crime Analytics 🚨")
st.sidebar.markdown("Uma ferramenta profissional para análise e investigação de padrões criminais em São Paulo.")

pagina_selecionada = st.sidebar.selectbox(
    "Selecione uma análise:",
    ["Página Inicial", "Dashboard Executivo", "Análise por Bairro", "Análise de Correlação", "Análise Temporal", "Investigação de Clusters"]
)

st.sidebar.markdown("---")
st.sidebar.info("Desenvolvido por Gemini, seu assistente de Ciência de Dados.")


# --- ESTRUTURA DAS PÁGINAS ---

# Página Inicial
if pagina_selecionada == "Página Inicial":
    st.title("Bem-vindo ao SP Crime Analytics")
    st.subheader("Transformando Dados em Inteligência para a Segurança Pública")
    st.markdown("""
    Esta plataforma interativa foi construída para explorar, visualizar e analisar dados de ocorrências criminais na cidade de São Paulo. 
    Nosso objetivo é fornecer uma ferramenta clara e poderosa para que analistas, gestores e o público em geral possam entender as dinâmicas da criminalidade na cidade.

    #### **O que você encontrará aqui?**
    - **Dashboard Executivo:** Uma visão macro da criminalidade com os principais indicadores e mapas de calor.
    - **Análise por Bairro:** Investigue a fundo a situação de um bairro específico de forma interativa.
    - **Análise de Correlação:** Entenda a relação estatística entre os itens roubados e as circunstâncias do crime.
    - **Análise Temporal:** Observe a evolução dos crimes no mapa ao longo do tempo.
    - **Investigação de Clusters:** Explore grupos de crimes com *modus operandi* similar, identificados por algoritmos de Machine Learning.

    Utilize o menu na barra lateral à esquerda para navegar entre as diferentes seções de análise.
    """)
    if df is not None:
        st.info(f"""
        **Visão Geral do Dataset:**
        - **Total de Registros Analisados:** {len(df):,}
        - **Período Coberto:** De {df['time'].min().strftime('%d/%m/%Y')} a {df['time'].max().strftime('%d/%m/%Y')}
        """, icon="📊")


# Dashboard Executivo
elif pagina_selecionada == "Dashboard Executivo":
    st.title("Dashboard Executivo da Criminalidade")
    
    if df is not None:
        st.markdown("*Esta seção apresenta os indicadores-chave de desempenho (KPIs) e visualizações gerais para um entendimento rápido do cenário criminal.*")

        total_ocorrencias = len(df)
        bairro_mais_comum = df['bairro'].mode()[0]
        hora_pico = df['hora'].mode()[0]
        principal_item = df[itens_colunas].sum().idxmax()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total de Ocorrências", f"{total_ocorrencias:,}", help="Número total de registros válidos no dataset.")
        col2.metric("Bairro com Mais Casos", bairro_mais_comum, help="O bairro que concentra o maior número de ocorrências.")
        col3.metric("Horário de Pico", f"{hora_pico}h", help="A hora do dia com maior frequência de crimes.")
        col4.metric("Principal Alvo", principal_item, help="O tipo de bem mais visado nos crimes registrados.")
        
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Distribuição de Crimes por Hora")
            st.markdown("*Este gráfico de barras mostra em quais horas do dia os crimes são mais frequentes, ajudando a identificar os períodos de maior risco.*")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.countplot(data=df, x='hora', ax=ax, palette='viridis')
            ax.set_xlabel("Hora do Dia")
            ax.set_ylabel("Contagem de Ocorrências")
            st.pyplot(fig)
        with col2:
            st.subheader("Distribuição por Dia da Semana")
            st.markdown("*Aqui vemos a contagem de crimes para cada dia da semana. Padrões podem indicar uma relação com a rotina da cidade (dias úteis vs. fins de semana).*")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.countplot(data=df, x='dia_semana', ax=ax, palette='plasma', order=df['dia_semana'].value_counts().index)
            ax.set_xlabel("Dia da Semana")
            ax.set_ylabel("Contagem")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
        st.markdown("---")
        
        st.subheader("Mapa Geográfico de Ocorrências (Hotspots)")
        st.markdown("*Este mapa interativo mostra a concentração geográfica dos crimes. A camada de calor (manchas vermelhas) indica áreas de alta incidência. Os círculos com números (clusters) mostram a contagem exata de ocorrências naquela região. Dê zoom para explorar.*")
        
        mapa_interativo = folium.Map(location=[-23.550520, -46.633308], zoom_start=11, tiles="cartodbdark_matter")
        
        heat_data = [[row['latitude'], row['longitude']] for _, row in df.iterrows()]
        HeatMap(heat_data, radius=15).add_to(mapa_interativo)
        
        marker_cluster = MarkerCluster().add_to(mapa_interativo)
        for _, row in df.sample(min(5000, len(df))).iterrows():
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=f"<b>{row['titulo']}</b>",
                icon=None
            ).add_to(marker_cluster)

        st_folium(mapa_interativo, width=1200, height=500, returned_objects=[])

# Análise por Bairro
elif pagina_selecionada == "Análise por Bairro":
    st.title("Análise Detalhada por Bairro")
    st.markdown("*Selecione um bairro na lista abaixo para filtrar todos os dados e visualizar um relatório específico para a região, incluindo seus próprios KPIs e mapas.*")
    
    bairros_unicos = sorted(df['bairro'].dropna().unique())
    bairro_selecionado = st.selectbox("Selecione um Bairro:", bairros_unicos)

    if bairro_selecionado:
        df_bairro = df[df['bairro'] == bairro_selecionado]
        st.header(f"Relatório para: {bairro_selecionado}")

        if not df_bairro.empty:
            principal_item_bairro = df_bairro[itens_colunas].sum().idxmax()
            hora_pico_bairro = f"{df_bairro['hora'].mode()[0]}h"
        else:
            principal_item_bairro = "N/A"
            hora_pico_bairro = "N/A"

        col1, col2, col3 = st.columns(3)
        col1.metric("Total de Ocorrências", len(df_bairro))
        col2.metric("Horário de Pico", hora_pico_bairro)
        col3.metric("Principal Alvo no Bairro", principal_item_bairro)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Crimes por Hora (Local)")
            fig, ax = plt.subplots()
            sns.countplot(data=df_bairro, y='hora', ax=ax, palette='crest', orient='h')
            ax.set_xlabel("Contagem")
            ax.set_ylabel("Hora do Dia")
            st.pyplot(fig)
        with col2:
            st.subheader("Mapa de Ocorrências (Local)")
            if not df_bairro.empty:
                mapa_bairro = folium.Map(location=[df_bairro['latitude'].mean(), df_bairro['longitude'].mean()], zoom_start=14)
                for _, row in df_bairro.iterrows():
                    folium.Marker(
                        location=[row['latitude'], row['longitude']],
                        popup=f"<b>{row['titulo']}</b><br>{row['time'].strftime('%d/%m/%Y %H:%M')}"
                    ).add_to(mapa_bairro)
                st_folium(mapa_bairro, width=800, height=400, returned_objects=[])
            else:
                st.write("Não há dados geográficos para exibir.")

# Análise de Correlação
elif pagina_selecionada == "Análise de Correlação":
    st.title("Análise de Correlação entre Variáveis")
    st.markdown("*Esta seção explora a relação estatística entre diferentes aspectos dos crimes, como o horário, o valor do prejuízo e os itens levados.*")
    
    st.info("""
    **Como interpretar este gráfico?**
    - A **Matriz de Correlação** mostra como as variáveis se movem em conjunto.
    - **Cores quentes (próximas de +1.0):** Indicam uma correlação positiva forte. Quando um item é roubado, o outro também tende a ser. Ex: `Dinheiro` e `Carteira`.
    - **Cores frias (próximas de -1.0):** Indicam uma correlação negativa. Quando um aumenta, o outro diminui.
    - **Cores neutras (próximas de 0):** Indicam que não há uma relação linear clara entre as variáveis.
    """, icon="💡")

    colunas_corr = ['hora', 'valor_prejuizo'] + itens_colunas
    matriz_corr = df[colunas_corr].corr()

    fig, ax = plt.subplots(figsize=(18, 15))
    sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
    ax.set_title('Matriz de Correlação', fontsize=18)
    st.pyplot(fig)


# Análise Temporal
elif pagina_selecionada == "Análise Temporal":
    st.title("Evolução Temporal dos Crimes")
    st.markdown("*Utilize o controle deslizante de tempo na parte inferior do mapa para navegar mês a mês. Este mapa **não é cumulativo**: ele mostra um 'snapshot' de cada mês, permitindo observar a dinâmica e o deslocamento da mancha criminal.*")

    df_mapa_temporal = df.sort_values('time').tail(5000)
    features = []
    for _, row in df_mapa_temporal.iterrows():
        feature = {
            'type': 'Feature',
            'geometry': {'type': 'Point', 'coordinates': [row['longitude'], row['latitude']]},
            'properties': {
                'time': row['time'].strftime('%Y-%m-%d'),
                'popup': f"<b>{row['titulo']}</b><br>Data: {row['time'].strftime('%d/%m/%Y')}",
                'icon': 'circle', 'iconstyle': {'fillColor': '#FF4B4B', 'fillOpacity': 0.8, 'stroke': 'false', 'radius': 6}
            }
        }
        features.append(feature)

    mapa_temporal = folium.Map(location=[-23.550520, -46.633308], zoom_start=11, tiles="cartodbdark_matter")
    folium.plugins.TimestampedGeoJson(
        {'type': 'FeatureCollection', 'features': features},
        period='P1M', duration='P1M', add_last_point=True
    ).add_to(mapa_temporal)
    st_folium(mapa_temporal, width=1200, height=600, returned_objects=[])


# Investigação de Clusters
elif pagina_selecionada == "Investigação de Clusters":
    st.title("Investigação de Clusters de Crimes")
    st.info("""
    **O que é um Cluster?** Um 'cluster' é um grupo de crimes que um algoritmo de Machine Learning identificou como sendo muito semelhantes entre si, com base na localização, horário e, principalmente, no *modus operandi* descrito no texto da ocorrência. Analisar um cluster é como investigar um padrão de atuação de um mesmo indivíduo ou grupo criminoso.
    """, icon="🤖")

    st.warning("A análise abaixo é uma **simulação** dos resultados do modelo de clusterização. Os grupos foram gerados aleatoriamente para fins de demonstração da interface.")

    if 'cluster_kmeans' not in df.columns:
        df['cluster_kmeans'] = pd.Series(
            data=np.random.randint(0, 10, len(df)),
            index=df.index
        )
    
    cluster_selecionado = st.slider("Selecione o Cluster para Análise:", 0, 9, 0, help="Arraste para explorar os diferentes perfis de crime agrupados pelo algoritmo.")
    df_cluster = df[df['cluster_kmeans'] == cluster_selecionado]

    st.header(f"Análise do Perfil do Cluster {cluster_selecionado}")

    with st.expander("Ver Estatísticas Detalhadas do Cluster", expanded=True):
        if not df_cluster.empty:
            bairro_comum_cluster = df_cluster['bairro'].mode()[0]
            hora_media_cluster = int(df_cluster['hora'].mean())
            principal_alvo_cluster = df_cluster[itens_colunas].sum().idxmax()
        else:
            bairro_comum_cluster, hora_media_cluster, principal_alvo_cluster = "N/A", "N/A", "N/A"

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Nº de Ocorrências no Cluster", len(df_cluster))
        col2.metric("Bairro Principal do Cluster", bairro_comum_cluster)
        col3.metric("Horário Médio de Atuação", f"~{hora_media_cluster}h")
        col4.metric("Principal Alvo do Grupo", principal_alvo_cluster)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Nuvem de Palavras do Modus Operandi")
        st.markdown("*As palavras maiores representam os termos mais frequentes nas descrições dos crimes deste grupo, revelando o método de atuação.*")
        texto_completo_cluster = ' '.join(df_cluster['titulo'].fillna('') + ' ' + df_cluster['descricao'].fillna(''))
        if texto_completo_cluster.strip():
            wordcloud = WordCloud(width=400, height=300, background_color='white', colormap='Reds').generate(texto_completo_cluster)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.write("Não há texto suficiente para gerar a nuvem de palavras.")
    
    with col2:
        st.subheader("Grafo de Conexões Temporais")
        st.markdown("*Este grafo conecta crimes do mesmo cluster que ocorreram em um intervalo de até 3 dias. Linhas conectando os pontos (crimes) sugerem uma possível 'série' de ações do mesmo grupo em um curto período.*")
        if len(df_cluster) > 1:
            G = nx.Graph()
            df_sample = df_cluster.sample(n=min(30, len(df_cluster)))
            for _, row in df_sample.iterrows():
                G.add_node(row['id'])
            
            ids = list(df_sample['id'])
            times = list(df_sample['time'])
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    if abs((times[i] - times[j]).days) <= 3:
                        G.add_edge(ids[i], ids[j])

            fig, ax = plt.subplots()
            pos = nx.spring_layout(G, seed=42)
            nx.draw(G, pos, ax=ax, with_labels=False, node_size=100, node_color='#FF4B4B', edge_color='gray')
            st.pyplot(fig)
        else:
            st.write("Não há ocorrências suficientes para gerar uma rede de conexões.")
