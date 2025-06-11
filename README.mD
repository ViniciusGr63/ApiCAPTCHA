# Captcha Api

# Captcha API

[![Documentação](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS3bXEBwo5YWH0tIyNIQp8ldcdCkhDix5mofA&s)](https://apicaptcha-xzmu.onrender.com/)

## Sobre o Projeto

A **Captcha API** é uma aplicação backend que oferece proteção contra bots automatizados. Ela foi projetada para proteger sites e aplicativos, impedindo que ações indesejadas sejam realizadas por scripts maliciosos. A API fornece endpoints RESTful para operações de leitura e processamento de imagens, permitindo que estas sejam classificadas em três tipos de formas geométricas: **triângulo**, **círculo** ou **quadrado**.

## Como Funciona?

- **Requisição do Captcha**: Ao acessar um formulário, fazer login ou interagir com algum recurso sensível, a API gera um desafio Captcha para o usuário.
- **Resposta do Usuário**: O usuário realiza a tarefa proposta, que consiste em identificar uma imagem de uma forma geométrica específica.
- **Validação da Resposta**: A API verifica se a resposta fornecida corresponde corretamente à forma que ela aprendeu a identificar (triângulo, círculo ou quadrado).

## Recursos e Funcionalidades

- **Reconhecimento**: Reconhece diferentes imagens a partir da generalização de um modelo.
- **Processamento**: Processa diferentes tipos de imagem para um padrão de reconhecimento.
- **Classificação**: Classifica as imagens entre as diferentes formas geométricas já conhecidas.

## URL Base

A API está disponível na seguinte URL:

```
https://apicaptcha-xzmu.onrender.com
```

## Tecnologias Utilizadas

- **Python**
- **Flask**
- **Prisma**
- **Banco de Dados Relacional** (PostgreSQL)
- **Render** (opcional para deployment)



## Como Executar Localmente

1. Clone o repositório:
   ```bash
   git clone https://github.com/ViniciusGr63/ApiCAPTCHA.git
   cd api
   ```
   
2.  Compile e execute a aplicação:
   ```bash
   source venv/bin/activate 
   gunicorn "api.captcha_app:create_app()"
   ```
   
4. Acesse a API em `http://localhost:8080`

## Endpoint Principal

## Recognize

| Método | Endpoint             | Descrição                             |
|--------|----------------------|---------------------------------------|
| POST   | `/captcha/recognize` | Classifica uma imagem                 |



## Exemplo de Uso

### Classificar uma imagem

**Requisição:**

``` http
POST /captcha/recognize
```

**Resposta:**

``` json
[
  {
    "shape": "squares",
  }
]
```
**Requisição:**

``` http
POST /captcha/recognize
```

**Resposta:**

``` json
[
  {
    "shape": "triangles",
  }
]
```
**Requisição:**

``` http
POST /captcha/recognize
```

**Resposta:**

``` json
[
  {
    "shape": "circles",
  }
]
```
