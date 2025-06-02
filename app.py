import gradio as gr
from ultralytics import YOLO
import os

model = YOLO("best.pt")

def detectar(imagen):
    resultado = model(imagen)[0]
    imagen_anotada = resultado.plot()
    return imagen_anotada

demo = gr.Interface(fn=detectar, inputs="image", outputs="image", title="Detección de Daños en Carreteras")

port = int(os.environ.get("PORT", 7860))  # Render asigna el puerto en la variable PORT

demo.launch(server_name="0.0.0.0", server_port=port)

