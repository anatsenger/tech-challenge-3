# Corrida Genética
Tech Chalenge 3:
Projeto de Pós-Graduação da FIAP — Disciplina de OpenAI

# Visão geral

Este repositório demonstra um fluxo completo de fine-tuning de um foundation model para responder perguntas sobre títulos de produtos, retornando descrições aprendidas a partir do dataset AmazonTitles-1.3MM (colunas title e content). O projeto segue as etapas pedidas no enunciado do Tech Challenge: preparar o dataset, testar o modelo base, treinar (fine-tuning) e testar o modelo treinado com prompts de entrada.

# Estrutura

finetuning_summarizer.ipynb — notebook com todo o pipeline:

1. Preparação do ambiente: imports, montagem do Google Drive e variáveis de ambiente.

2. Leitura e tratamento da base: leitura de trn.json, filtragem/limpeza e geração de trn_tratado.json contendo title e content.

3. Modelo base (pré-treino): carregamento via unsloth.FastLanguageModel e teste antes do fine-tuning.

4. Fine-tuning (SFT): formatação de prompts, datasets.load_dataset do JSON tratado, trl.SFTTrainer + transformers.TrainingArguments.

5. Testes pós-treino: geração com o modelo já ajustado (comparativo com a etapa 3).

# Requisitos

Python 3.10+
GPU com CUDA (ex.: Google Colab com T4/A100)
Bibliotecas principais:

unsloth (carregamento e aceleração 4-bit)
transformers, trl, datasets, torch
(opcional) google.colab para montar o Drive

# Instalação (exemplo Colab)
pip install -q unsloth transformers trl datasets accelerate bitsandbytes

# Dataset

Fonte: AmazonTitles-1.3MM

Arquivo: trn.json (usar colunas title e content)

Tratamento: o notebook filtra registros com content válido e salva trn_tratado.json com { "title": ..., "content": ... }.

# Modelo e Prompt

Modelo base: unsloth/llama-3-8b-bnb-4bit (carregado via FastLanguageModel.from_pretrained em 4-bit).

Prompt de instrução (resumo do notebook):

Instrução: “Descreva o produto de [INPUT]”

Input: recebe o título (title)

Response: esperado o texto de descrição (content)

Geração de teste: uso de TextStreamer e model.generate(...) antes e depois do treinamento.

# Fine-Tuning (SFT)

Formatação: função formatting_prompts_func concatena instrução + entrada (title) + rótulo (content) + eos_token.

Dataset: datasets.load_dataset("json", data_files=AMAZON_TITLES_TRATADOS_PATH, split="train") + .map(...).

Trainer: trl.SFTTrainer(model=..., tokenizer=..., dataset_text_field="text", ...).

TrainingArguments (principais, do notebook):

per_device_train_batch_size = 2

gradient_accumulation_steps = 4

warmup_steps = 5

max_steps = 60

learning_rate = 2e-4

fp16/bf16 conforme suporte (is_bfloat16_supported())

optim = "adamw_8bit"

weight_decay = 0.01

lr_scheduler_type = "linear"

seed = 3407

output_dir = "outputs"

# Como rodar

1. Abra o notebook finetuning_summarizer.ipynb no Google Colab.

2. Monte seu Google Drive (célula já incluída) e ajuste:

  AMAZON_TITLES_PATH (caminho para trn.json)
  
  AMAZON_TITLES_TRATADOS_PATH (onde salvar trn_tratado.json)

3. Execute as células em ordem:

  Preparação do ambiente e imports
  
  Leitura/tratamento do dataset
  
  Teste do modelo base
  
  Fine-tuning (trainer.train())
  
  Teste pós-treino (comparar com o pré-treino)

4. Os artefatos de treino serão salvos em outputs/ (padrão do TrainingArguments.output_dir).
