import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg
import matplotlib

matplotlib.use("TkAgg")
fig = matplotlib.figure.Figure(figsize=(5, 4), dpi=100)
t = np.arange(0, 3, 0.01)
fig.add_subplot(111).plot(t, 2 * np.sin(2 * np.pi * t))


def draw_figure(canvas, figure):
    tkcanvas = FigureCanvasTkAgg(figure, canvas)
    tkcanvas.draw()
    tkcanvas.get_tk_widget().pack(side="top", fill="both", expand=1)
    return tkcanvas


layout = [[sg.Text("Plot test")], [sg.Canvas(key="-CANVAS-")], [sg.Button("Ok")]] # 
window = sg.Window(
    "Matplotlib In PySimpleGUI",
    layout,
    size=(150, 150),
    finalize=True,
    element_justification="center",
    font="Helvetica 18",
)

# add the plot to the window
tkcanvas = draw_figure(window["-CANVAS-"].TKCanvas, fig)
event, values = window.read()
window.close()
