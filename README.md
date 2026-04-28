# 💬 Análise de Sentimentos em Avaliações de Clientes Bancários com NLP

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-NLTK%20%7C%20TF--IDF-purple)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-SVM%20%7C%20RF%20%7C%20LR-orange)
![Status](https://img.shields.io/badge/Status-Concluído-brightgreen)
![Idioma](https://img.shields.io/badge/Dataset-Português%20🇧🇷-green)

> Pipeline completo de NLP para classificar automaticamente reviews de clientes bancários em Positivo, Neutro ou Negativo — do texto bruto ao modelo em produção.

---

## 📋 Sumário

- [O Problema de Negócio](#-o-problema-de-negócio)
- [Dataset](#-dataset)
- [Pipeline de NLP](#-pipeline-de-nlp)
- [Resultados](#-resultados)
- [Interpretabilidade](#-interpretabilidade)
- [Como Executar](#-como-executar)
- [Estrutura do Repositório](#-estrutura-do-repositório)
- [Tecnologias](#-tecnologias)

---

## 💡 O Problema de Negócio

Bancos e fintechs recebem **milhares de avaliações por dia** no Google Play, App Store e Reclame Aqui. Ler manualmente cada uma é impossível em escala.

Um modelo de NLP que classifica reviews automaticamente permite:

| Sentimento | Ação Imediata |
|---|---|
| 😡 **Negativo** | Alerta para o time de CX — resposta em minutos |
| 😐 **Neutro** | Monitoramento e priorização de backlog |
| 😊 **Positivo** | Identificar o que está funcionando bem |

> **Resultado prático:** um analista que levaria 8h para classificar 500 reviews passa a fazer isso em segundos — com consistência e sem fadiga de decisão.

---

## 📊 Dataset

- **999 avaliações** de clientes de bancos digitais brasileiros (Nubank, Bradesco, Itaú, Santander, C6 Bank, Inter, Neon, Next, PicPay, Banco do Brasil)
- **Idioma:** Português brasileiro 🇧🇷
- **3 classes balanceadas:** positivo (333) · neutro (333) · negativo (333)
- **4 colunas:** `review`, `sentimento`, `nota` (1-5), `banco`

---

## 🔬 Pipeline de NLP

```
Texto Bruto → Pré-processamento → TF-IDF → Modelo → Avaliação → Interpretabilidade
```

### 1. Pré-processamento
- Lowercase, remoção de URLs e menções
- Remoção de pontuação e números
- Remoção de **stopwords em português** (NLTK)
- **Stemming com RSLP** (Stemmer específico para português)
  - Ex: "transferências" → "transfer" | "cobranças" → "cobr"

### 2. Vetorização — TF-IDF
- **TF (Term Frequency):** frequência da palavra no review
- **IDF (Inverse Document Frequency):** penaliza palavras comuns em todos os reviews
- `ngram_range=(1,2)`: captura palavras isoladas ("ruim") e pares ("atendimento ruim")
- `max_features=5000`: as 5.000 features mais informativas
- `sublinear_tf=True`: suavização logarítmica das frequências

### 3. Modelos Treinados

| Modelo | Justificativa |
|---|---|
| **Regressão Logística** | Baseline — simples e interpretável |
| **Random Forest** | Captura relações não-lineares no vocabulário |
| **SVM Linear** | Estado da arte para text classification com TF-IDF |

> **Pipeline scikit-learn** garante que o TF-IDF seja ajustado **apenas nos dados de treino**, prevenindo data leakage.

---

## 📈 Resultados

| Modelo | F1-Score (CV 5-fold) | F1-Score (Teste) | Acurácia |
|---|---|---|---|
| Regressão Logística | ~0.88 | ~0.87 | ~0.87 |
| Random Forest | ~0.85 | ~0.84 | ~0.84 |
| **SVM Linear** | **~0.89** | **~0.88** | **~0.88** |

**SVM Linear** é o modelo escolhido — padrão para classificação de texto em alta dimensão.

**Por que o Neutro é o mais difícil?**  
Reviews neutros misturam elementos positivos e negativos no mesmo texto (*"funciona bem, mas poderia melhorar"*). É ambiguidade linguística genuína — 88% de acurácia nesse cenário é um resultado sólido.

---

## 🔍 Interpretabilidade

Ao contrário de modelos caixa-preta, TF-IDF + SVM linear permitem inspecionar exatamente **quais palavras mais pesam em cada classificação**:

| Sentimento | Principais palavras preditivas |
|---|---|
| 😊 Positivo | excelente, rápido, recomendo, ótimo, incrível, funciona, simples |
| 😡 Negativo | horrível, péssimo, absurdo, cobrar, bloquearam, demora, problema |
| 😐 Neutro | razoável, mediano, básico, poderia, funciona, aguardando |

---

## 🚀 Como Executar

```bash
git clone https://github.com/bellaizaoliveira/sentiment-analysis-banking.git
cd sentiment-analysis-banking
pip install -r requirements.txt
jupyter notebook notebooks/sentiment_analysis_banking.ipynb
```

O dataset já está em `data/avaliacoes_bancarias.csv` — basta rodar as células em ordem.

---

## 📁 Estrutura do Repositório

```
sentiment-analysis-banking/
│
├── data/
│   └── avaliacoes_bancarias.csv        # 999 reviews de clientes bancários (PT-BR)
│
├── notebooks/
│   └── sentiment_analysis_banking.ipynb # Pipeline completo NLP → modelo → insights
│
├── images/                              # Gráficos gerados pelo notebook
│   ├── 01_eda_overview.png
│   ├── 02_top_words.png
│   ├── 03_model_comparison.png
│   ├── 04_confusion_matrix.png
│   └── 05_top_features.png
│
├── requirements.txt
└── README.md
```

---

## 🛠 Tecnologias

| Tecnologia | Uso |
|---|---|
| **Python 3.9+** | Linguagem principal |
| **NLTK** | Stopwords PT + Stemmer RSLP (português) |
| **Scikit-learn** | TF-IDF, Pipeline, SVM, RF, LR, métricas |
| **Pandas / NumPy** | Manipulação de dados |
| **Matplotlib / Seaborn** | Visualizações |
| **Jupyter Notebook** | Ambiente interativo |

---

## 👩‍💻 Autora

**Izabella da Silva Oliveira**  
Cientista de Dados | IA Generativa & Machine Learning

[![LinkedIn](https://img.shields.io/badge/LinkedIn-bellaiza-blue?logo=linkedin)](https://linkedin.com/in/bellaiza)
[![GitHub](https://img.shields.io/badge/GitHub-bellaizaoliveira-black?logo=github)](https://github.com/bellaizaoliveira)

---

*Projeto desenvolvido com fins educacionais e de portfólio.*
