import gradio as gr
from ultralytics import YOLO
import cv2

model = YOLO("best.pt")

def detectar(imagen):
    resultado = model(imagen)[0]
    imagen_anotada = resultado.plot()
    return imagen_anotada

demo = gr.Interface(fn=detectar, inputs="image", outputs="image", title="Detección de Daños en Carreteras")
demo.launch()
