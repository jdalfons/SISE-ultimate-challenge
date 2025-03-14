---
title: Sise Challenge Emotional Report
emoji: 🎤
colorFrom: yellow
colorTo: green
sdk: docker
pinned: false
---

# SISE Ultimate Challenge - Emotional Report



Welcome to **Emotional Report**! This AI-powered application lets users send or record an audio clip 📢, analyzing their emotional state based on vocal tone and speed. The AI predicts whether the emotion falls into one of three categories: **Anger (Colère) 😡, Joy (Joie) 😃, or Neutral (Neutre) 😐**.

Using **Wav2Vec**, a pre-trained AI model, the app not only detects emotions but also attempts to transcribe the speech into text. 🧠🎙️

---

## 🎬 Fun Fact

The name **Emotional Report** is inspired by the movie *Minority Report*, where AI predicts crimes before they happen! 🔮
This challenge is the **Ultimate Challenge** for Master SISE students. 🏆

---

## 👀 Overview

This project features a **Streamlit-based dashboard** 📊 that helps analyze security logs, data trends, and apply machine learning models.

### ✨ Features

✅ **Home** - Overview of the challenge 🏠
✅ **Analytics** - Visualize & analyze security logs and data trends 📈
✅ **Machine Learning** - Train & evaluate ML models 🤖

---

## 🚀 Installation Guide

### 🔧 Local Setup

Follow these steps to run the project locally:

1. **Clone the repository:**
   ```sh
   git clone https://github.com/jdalfons/sise-ultimate-challenge.git
   cd sise-ultimate-challenge
   ```
2. **Create and activate a virtual environment:**
   ```sh
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
4. **Run the Streamlit application:**
   ```sh
   streamlit run app.py
   ```

### 🐳 Docker Setup

1. **Build the Docker image:**
   ```sh
   docker build -t sise-ultimate-challenge .
   ```
2. **Run the container:**
   ```sh
   docker run -p 7860:7860 sise-ultimate-challenge
   ```

---

## ⚙️ Technical Details

- 🐍 **Python 3.12**
- 🎨 **Streamlit**
- 🎙️ **Wav2Vec2**

---

## 🤝 Contributors

- [Cyril KOCAB](https://github.com/Cyr-CK) 👨‍💻
- [Falonne KPAMEGAN](https://github.com/marinaKpamegan) 👩‍💻
- [Juan ALFONSO](https://github.com/jdalfons) 🎤
- [Nancy RANDRIAMIARIJAONA](https://github.com/yminanc) 🔍

🔥 *Join us in making AI-powered emotion detection awesome!*

