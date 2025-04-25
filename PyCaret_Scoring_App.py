import os
import pickle
import pandas as pd
import streamlit as st
import requests
from io import BytesIO
from pycaret.classification import predict_model

# Utilitário para converter para Excel
@st.cache_data
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

# Função para fazer download do modelo a partir da URL
def load_model_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Verifica se a requisição foi bem-sucedida
        model = pickle.loads(response.content)
        return model
    except Exception as e:
        st.error(f"❌ Erro ao carregar o modelo da URL: {e}")
        return None

# Função principal do Streamlit
def main():
    # Configuração da página
    st.set_page_config(
        page_title="🧠 PyCaret Scoring App",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🧠 Escoragem com Modelo Treinado no PyCaret")
    st.markdown("Faça upload de um arquivo .ftr ou .csv, e o modelo aplicará a escoragem automaticamente.")
    st.markdown("---")

    # Upload de arquivo
    st.sidebar.header("📁 Upload do Arquivo")
    uploaded_file = st.sidebar.file_uploader("Selecione um arquivo de entrada", type=['ftr', 'csv'])

    if uploaded_file is not None:
        st.success("✅ Arquivo carregado com sucesso!")
        
        # Leitura inteligente do arquivo
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_feather(uploaded_file)
        except Exception as e:
            st.error(f"❌ Erro ao ler o arquivo: {e}")
            return

        # Corrigir se estiver transposto
        if df.shape[1] > 1000 and df.shape[0] < 100:
            df = df.T.reset_index(drop=True)
            df.columns = df.iloc[0]
            df = df[1:].reset_index(drop=True)

        # Preview
        st.subheader("👀 Preview dos Dados")
        st.write(df.head())

        # Amostragem
        n_sample = st.slider("🔍 Tamanho da amostra para escoragem", 1000, min(50000, len(df)), 5000)
        df_sample = df.sample(n_sample).reset_index(drop=True)

        # Carregar modelo da URL
        st.markdown("---")
        st.subheader("⚙️ Escorando com o Modelo")
        model_url = "https://github.com/DanielaDF13/PyCaret-Scoring/blob/d791080b22555993fe26503013e7b8bf49dc2c44/data/model_final.pkl?raw=true"
        model = load_model_from_url(model_url)
        
        if model is None:
            return  # Se o modelo não for carregado, não continua a execução

        # Previsão
        try:
            predictions = predict_model(model, data=df_sample)
            st.success("✅ Previsão realizada com sucesso!")
            st.dataframe(predictions.head(10))
        except Exception as e:
            st.error(f"❌ Erro ao fazer a previsão: {e}")
            return

        # Baixar resultado
        st.markdown("---")
        st.subheader("📥 Baixar Resultados")
        df_xlsx = to_excel(predictions)
        st.download_button(
            label="📤 Baixar arquivo Excel com Previsões",
            data=df_xlsx,
            file_name="previsoes.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


if __name__ == "__main__":
    main()
