import time
import argparse
import time
import os
import json
import io
import pathlib
import warnings
from typing import TYPE_CHECKING, Optional, Tuple, Union
import numpy as np
import torch
import tqdm
#import transcribe
#import model
import whisper
from whisper.tokenizer import Tokenizer, get_tokenizer

# Choose model to use by uncommenting
#modelName = "tiny.en"
#modelName = "base.en"
#modelName = "small.en"
#modelName = "medium.en"
modelName = "large-v2"
timestamp = int(time.time())
file_name = 'f'
# Other Variables
exportTimestampData = True # (bool) Whether to export the segment data to a json file. Will include word level timestamps if word_timestamps is True.
outputFolder = "Output"

#  ----- Select variables for transcribe method  -----
audio = whisper.load_audio("oh.mp3")
verbose = False # (bool): Whether to display the text being decoded to the console. If True, displays all the details, If False, displays minimal details. If None, does not display anything
language="fr" # Language of audio file
word_timestamps=True # (bool): Extract word-level timestamps using the cross-attention pattern and dynamic time warping, and include the timestamps for each word in each segment.
#initial_prompt="" # (optional str): Optional text to provide as a prompt for the first window. This can be used to provide, or "prompt-engineer" a context for transcription, e.g. custom vocabularies or proper nouns to make it more likely to predict those word correctly.

#  -------------------------------------------------------------------------
print(f"Using Model: {modelName}")
#filePath = input("D:\Documents\whisper\oh.mp3")
#filePath = filePath.strip("\"")
#if not os.path.exists(filePath):
	#print("Problem Getting File...")
	#input("Press Enter to Exit...")
	#exit()

# If output folder does not exist, create it
if not os.path.exists(outputFolder):
	os.makedirs(outputFolder)
	print("Created Output Folder.\n")

# Get filename stem using pathlib (filename without extension)
fileNameStem = pathlib.Path("file_name" + "timestamp").stem

resultFileName = f"{fileNameStem}.txt"
jsonFileName = f"{fileNameStem}.json"

model = whisper.load_model(modelName)
start = time.time()

#  ---------------------------------------------------
result = model.transcribe(audio=audio, language=language, word_timestamps=word_timestamps, verbose=verbose)
#  ---------------------------------------------------

end = time.time()
elapsed = float(end - start)
resultat = str(result["segments"])
# Save transcription text to file
print("\nWriting transcription to file...")
with open(os.path.join(outputFolder, resultFileName), "w", encoding="utf-8") as file:
    file.write(resultat)
print("Finished writing transcription file.")
# Save the segments data to json file
#if word_timestamps == True:
if exportTimestampData == True:
	print("\nWriting segment data to file...")
	with open(os.path.join(outputFolder, jsonFileName), "w", encoding="utf-8") as file:
		segmentsData = result["segments"]
		json.dump(segmentsData, file, indent=4)
	print("Finished writing segment data file.")

elapsedMinutes = str(round(elapsed/60, 2))
print(f"\nElapsed Time With {modelName} Model: {elapsedMinutes} Minutes")

input("Press Enter to exit...")
exit()

timestamp = int(time.time())

file_name = 'f'
#device = "cuda" if torch.cuda.is_available() else "cpu"
#print(torch.cuda.is_available())
#model = whisper.load_model("base").to(device)
#model = whisper.load_model("base", device = "cuda")
#audio = whisper.load_audio("oh.mp3")
#mel = whisper.log_mel_spectrogram(audio).to(model.device)
#options = whisper.DecodingOptions()
#result = model.transcribe(audio,language = "fr", verbose = True, return_timestamps = True)

#assert result["text"] == "".join([s["text"] for s in result["segments"]])
#transcription = result["text"].lower()

#tokenizer = get_tokenizer(model.is_multilingual)
#all_tokens = [t for s in result["segments"] for t in s["tokens"]]
#assert tokenizer.decode(all_tokens) == result["text"]
#assert tokenizer.decode_with_timestamps(all_tokens).startswith("<|0.00|>")
#resultat = tokenizer.decode_with_timestamps(all_tokens).startswith("<|0.00|>")

#f= open(f'{file_name}_{timestamp}.txt',"w+")
#f.write(result)
#f.close()
