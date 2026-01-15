import pandas as pd
import numpy as np
import gradio as gr

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# LOAD DATA
df = pd.read_csv("secondhandcar.csv")

# CLEAN TEXT DATA
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype(str).str.strip().str.lower()

# ENCODE CATEGORICAL COLUMNS
encoders = {}
for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

# TRAIN / TEST SPLIT
X = df.drop("selling_price", axis=1)
y = df["selling_price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TRAIN MODEL
model = RandomForestRegressor(
    n_estimators=120,
    max_depth=14,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# SAFE ENCODER
def encode(col, value):
    value = str(value).strip().lower()
    le = encoders[col]
    if value in le.classes_:
        return le.transform([value])[0]
    return le.transform([le.classes_[0]])[0]

# PREDICTION FUNCTION
def predict_price(
    name,
    year,
    km_driven,
    fuel,
    seller_type,
    transmission,
    owner
):
    input_df = pd.DataFrame([{
        "name": encode("name", name),
        "year": int(year),
        "km_driven": int(km_driven),
        "fuel": encode("fuel", fuel),
        "seller_type": encode("seller_type", seller_type),
        "transmission": encode("transmission", transmission),
        "owner": encode("owner", owner)
    }], columns=X.columns)

    price = model.predict(input_df)[0]
    return f"₹ {round(price, 2):,}"

# PROFESSIONAL UI (BRANDED)
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="cyan"
    )
) as app:

    # HEADER
    gr.Markdown(
        """
        <div style="text-align:center; padding:20px; background:linear-gradient(135deg,#0f2027,#203a43,#2c5364); border-radius:12px;">
            <h1 style="color:white;">🚗 Mukesh Car Predictor</h1>
            <p style="color:#d1e8ff; font-size:16px;">
                AI-powered Second-Hand Car Price Prediction using Random Forest
            </p>
        </div>
        """
    )

    gr.Markdown("### 🔧 Enter Car Details")

    with gr.Row():
        with gr.Column(scale=1):
            car_name = gr.Dropdown(
                encoders["name"].classes_.tolist(),
                label="Car Name"
            )

            year = gr.Slider(
                1990, 2026, step=1,
                label="Manufacturing Year"
            )

            km_driven = gr.Slider(
                0, 300000, step=1000,
                label="Kilometers Driven"
            )

            fuel = gr.Dropdown(
                encoders["fuel"].classes_.tolist(),
                label="Fuel Type"
            )

            seller_type = gr.Dropdown(
                encoders["seller_type"].classes_.tolist(),
                label="Seller Type"
            )

            transmission = gr.Radio(
                encoders["transmission"].classes_.tolist(),
                label="Transmission"
            )

            owner = gr.Dropdown(
                encoders["owner"].classes_.tolist(),
                label="Owner Type"
            )

            predict_btn = gr.Button("🔮 Predict Price", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### 💰 Estimated Resale Value")
            output = gr.Textbox(
                placeholder="Prediction will appear here...",
                label="Predicted Price",
                lines=2
            )

    predict_btn.click(
        fn=predict_price,
        inputs=[
            car_name,
            year,
            km_driven,
            fuel,
            seller_type,
            transmission,
            owner
        ],
        outputs=output
    )

    gr.Markdown(
        """
        ---
        **Built by Mukesh** | Random Forest ML Model  
        Ready for deployment on **Hugging Face Spaces**
        """
    )

app.launch(share=True)
