
# 🚗 Second-Hand Car Price Prediction

A **machine learning web application** that predicts the resale price of used cars based on key features.
Built using **Python + Random Forest**, powered by **Gradio UI**, and **deployed on Hugging Face Spaces** for live interaction.

---

## 🌐 Live Demo

🔗 **Hugging Face Space:**
👉 *https://tinyurl.com/car-price-predict*

Users can enter car details and instantly get a predicted resale price.

---

## 🧠 Project Overview

Pricing used cars accurately is a real-world regression problem.
This project uses historical car data to train a **Random Forest Regression model**, providing reliable predictions through an interactive web interface.

The app is deployed publicly using **Hugging Face Spaces**, making it accessible without local setup.

---

## ✨ Features

* 🚀 **Random Forest Regression model**
* 🎛️ **Gradio-based interactive UI**
* ☁️ **Deployed on Hugging Face Spaces**
* 📊 Real-time predictions
* 🧪 Clean preprocessing & model evaluation
* 📁 Simple and beginner-friendly project structure

---

## 🏗️ Tech Stack

* **Python**
* **Pandas, NumPy**
* **Scikit-learn**
* **Gradio**
* **Hugging Face Spaces**

---

## 📂 Project Structure

```
Secondhand_carprice_prediction-
│
├── app.py                 # Gradio app + model logic
├── model.pkl              # Trained Random Forest model
├── data.csv               # Dataset used for training
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

---

## ⚙️ How It Works

1. Load and preprocess second-hand car dataset
2. Train a **Random Forest Regressor**
3. Save the trained model
4. Build a **Gradio UI** for user input
5. Deploy the app on **Hugging Face Spaces**
6. Users enter car details → get predicted price instantly

---

## 🧪 Input Features

* Car Name / Brand
* Manufacturing Year
* Kilometers Driven
* Fuel Type
* Seller Type
* Transmission
* Owner Type

---

## 📈 Model Used

* **Random Forest Regressor**
* Handles non-linear relationships well
* Robust against overfitting
* Performs better than basic linear models for this dataset

---

## 🖥️ Run Locally (Optional)

```bash
git clone https://github.com/mukesh-kumar-git/Secondhand_carprice_prediction-
cd Secondhand_carprice_prediction-
pip install -r requirements.txt
python app.py
```

The Gradio interface will open in your browser automatically.

---

## 🚀 Deployment

This project is deployed using **Hugging Face Spaces** with:

* `app.py` as the entry point
* `requirements.txt` for dependency management
* Gradio as the frontend framework

---

## 🔮 Future Improvements

* Add more advanced feature engineering
* Improve UI with custom themes
* Add model comparison (XGBoost, Gradient Boosting)
* Include confidence intervals for predictions
* Enable CSV upload for bulk predictions

---

## 📜 License

This project is open-source and free to use for learning and experimentation.

---
## 🚀 Live Demo Preview

Below is the Gradio-based interface deployed on Hugging Face Spaces, allowing users to input car details and get real-time price predictions.

![Second-Hand Car Price Prediction Gradio UI](Screenshots/Gradio_UI.png)

---
## 🙌 Acknowledgements

* Scikit-learn documentation
* Gradio framework
* Hugging Face Spaces
