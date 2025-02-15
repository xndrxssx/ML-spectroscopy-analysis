# Análise de Espectroscopia com Machine Learning

## Descrição do Projeto
Este projeto tem como objetivo analisar dados de espectroscopia de uma variedade de uva utilizando técnicas de machine learning para identificar padrões e prever atributos de qualidade como SST (Sólidos Solúveis Totais), AT (Acidez Total), pH, Firmeza e UBS (Umidade da Baga Seca). O pipeline é organizado em pastas específicas para cada etapa do processo incluindo, pré-processamento dos dados, redução de dimensionalidade com PCA, seleção de amostras com o método Kennard-Stone e aplicação de modelos como PCR, PLSR, Random Forest, SVR e MLP.

## Funcionalidades
- **Pré-processamento**: Normalização e padronização dos dados espectroscópicos.
- **Redução de Dimensionalidade**: Aplicação de PCA para reduzir o número de features.
- **Seleção de Amostras**: Uso do método Kennard-Stone para dividir os dados em conjuntos de calibração e validação.
- **Modelos de Machine Learning**:
  - Principal Component Regression (PCR)
  - Partial Least Squares Regression (PLSR)
  - Random Forest Regression (RFR)
  - Support Vector Regression (SVR)
  - Multi-Layer Perceptron Regression (MLPR)

## Como Usar
1. **Clone o repositório**:
   ```bash
   git clone https://github.com/xndrxssx/ML-spectroscopy-analysis.git
    ```
2. **Instale as dependências**
   ```bash
   pip install -r requirements.txt
    ```
3. **Abra o arquivo Jupyter de cada pasta e execute para visualizar os resultados.**

## Estrutura do projeto

/ML-spectroscopy-analysis
│
├── /Data                                       # Dados brutos
├── /Principal Components Analysis              # Análise de Componentes Principais (PCA)
├── /Pre-processing                             # Pré-processamento dos dados
├── /Principal Components Regression            # Regressão por Componentes Principais (PCR)
├── /Processed                                  # Dados processados
├── .gitignore                                  # Arquivo para ignorar arquivos desnecessários
├── README.md                                   # Documentação do projeto
└── requirements.txt                            # Dependências do projeto

## Licença

Este projeto está licenciado sob a [MIT License](https://choosealicense.com/licenses/mit/).

## Contato
Se tiver dúvidas ou sugestões, entre em contato:

Nome: [@Andressa](https://www.linkedin.com/in/andressa-carvalho-6b09b2312/)

Email: [acarvalho0710@gmail.com]

GitHub: xndrxssx