import gradio as gr

#model
#nom fichier de sortie
#dossier de sortie
#fichier qu'on analyse
#langue
import whisper
#def load_models(model_names):
 #  model = whisper.load_model(modelName)

  # return model


model_names = [
    "tiny",
    "base",
    "small",
    "medium",
    "large",
]

#models = {model_names: load_model(modelName) for model_names in model_names}

#frame1, frame2, times_to_interpolate, 
def predict(model_names, fichier):
    model = whisper.load_model(model_names)

   # frame1 = resize(960, frame1)
    #frame2 = resize(960, frame2)
#
 #   frame1.save("test1.png")
  #  frame2.save("test2.png")

   # resize_img("test1.png", "test2.png")
    #input_frames = ["test1.png", "resized_img2.png"]

   # frames = list(
    #    util.interpolate_recursively_from_files(
     #       input_frames, times_to_interpolate, model))

   # mediapy.write_video("out.mp4", frames, fps=30)
    return model_names


title = "Whisper"
description = "Transcription de paroles vers texte"
article = "<p style='text-align: center'><a href='https://github.com/openai/whisper>"

gr.Interface(
    predict,
    [
        gr.inputs.Dropdown(choices=model_names, default=model_names[0]),
        gr.inputs.Audio(type="filepath"), 
       # gr.inputs.Image(type='filepath'),
      #  gr.inputs.Slider(minimum=2, maximum=100, step=1),
        
    ],
    outputs = ["text"],
    title=title,
    description=description,
    article=article,
).launch(enable_queue=True)


print(fichier)
print(model)
