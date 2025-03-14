```yaml
---
title: Sise Challenge
emoji: 🎤
colorFrom: yellow
colorTo: green
sdk: docker
pinned: false
---
```
# SISE  Ultimate Challenge
![Logo du Ultimate Challenge SISE](img/logo_01.png)

Ceci est le Ultimate Challenge pour le Master SISE.

## Aperçu

Ce projet est un tableau de bord basé sur Streamlit pour analyser les journaux de sécurité, les tendances des données et appliquer des modèles d'apprentissage automatique.

## Fonctionnalités

- Accueil : Vue d'ensemble du défi
- Analytique : Visualiser et analyser les journaux de sécurité et les tendances des données
- Apprentissage Automatique : Entraîner et évaluer des modèles d'apprentissage automatique

## Installation

### Locale
Pour exécuter ce projet localement, suivez ces étapes :

1. Clonez le dépôt :
    ```sh
    git clone https://github.com/jdalfons/sise-ultimate-challenge.git
    cd sise-ultimate-challenge
    ```

2. Créez un environnement virtuel et activez-le :
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Installez les dépendances requises :
    ```sh
    pip install -r requirements.txt
    ```

### Docker
1. Construisez l'image Docker :
    ```sh
    docker build -t sise-ultimate-challenge .
    ```

2. Exécutez le conteneur Docker :
    ```sh
    docker run -p 7860:7860 sise-ultimate-challenge
    ```
## Utilisation

Pour démarrer l'application Streamlit, exécutez la commande suivante :
```sh
streamlit run app.py
```