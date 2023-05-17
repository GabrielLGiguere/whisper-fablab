import gradio as gr

#model
#nom fichier de sortie
#dossier de sortie
#fichier qu'on analyse
#langue

def load_models(model_names):
   model = whisper.load_model(model_names)

   return model


model_names = [
    "tiny",
    "base",
    "small",
    "medium",
    "large",
]

models = {model_names: load_model(model_names) for model_names in model_names}


def predict(frame1, frame2, times_to_interpolate, model_names):
    model = models[model_names]

    frame1 = resize(960, frame1)
    frame2 = resize(960, frame2)

    frame1.save("test1.png")
    frame2.save("test2.png")

    resize_img("test1.png", "test2.png")
    input_frames = ["test1.png", "resized_img2.png"]

    frames = list(
        util.interpolate_recursively_from_files(
            input_frames, times_to_interpolate, model))

    mediapy.write_video("out.mp4", frames, fps=30)
    return "out.mp4"


title = "frame-interpolation"
description = "Gradio demo for FILM: Frame Interpolation for Large Scen"
article = "<p style='text-align: center'><a href='https://film-net.gith>"
examples = [
    ['cat3.jpeg', 'cat4.jpeg', 2, model_names[0]],
    ['cat1.jpeg', 'cat2.jpeg', 2, model_names[1]],
]

gr.Interface(
    predict,
    [
        gr.inputs.Image(type='filepath'),
        gr.inputs.Image(type='filepath'),
        gr.inputs.Slider(minimum=2, maximum=100, step=1),
        gr.inputs.Dropdown(choices=model_names, default=model_names[0])
    ],
    "playable_video",
    title=title,
    description=description,
    article=article,
    examples=examples
).launch(enable_queue=True)
