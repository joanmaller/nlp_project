import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import sys

# Default files 
path_to_file = 'amy-winehouse.txt'
path_to_model = './amy_winehouse.h5'

# Dataset as first argument
if len(sys.argv) > 1:
  path_to_file = sys.argv[1]
# Model as second argument
if len(sys.argv) > 2:
  path_to_model = sys.argv[2]


# Open model and dataset 
model = tf.keras.models.load_model(path_to_model) 
 
#Get name of file 
name = path_to_file.split('.')[0]
#print(name)
data= open(path_to_file).read()
corpus = data.lower().split('\n')

#Tokenize data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1
word_index = tokenizer.word_index 
index_word = {index:word for word, index in tokenizer.word_index.items()}

n_gram_sequences = []
for line in corpus:
  token_list = tokenizer.texts_to_sequences([line])[0]
  for i in range(1, len(token_list)):
    n_gram_sequences.append(token_list[:i+1])
#print(np.shape(n_gram_sequences))

#for seq in n_gram_sequences:
#  print ([index_word[w] for w in seq])

# pad sequences
max_len = max([len(seq) for seq in n_gram_sequences])

#print(max_sequence_len, total_words)
n_gram_sequences = np.array(pad_sequences(n_gram_sequences, padding='pre', maxlen=max_len))


# Generate next words with an initial prompt
def predict_n_words(prompt, n_words):
  for _ in range(n_words):
    token_list = tokenizer.texts_to_sequences([prompt])[0]
    token_list = pad_sequences([token_list], padding='pre', maxlen=max_len-1,)
    predicted = np.argmax( model.predict(token_list), axis = 1)
    prompt += " " + index_word[predicted[0]]
  return prompt


n_words = 100

prompt = "the girl from ipanema"
prompt2 = "A walk through Cagliari, "
prompts = [prompt, prompt2]

generated_text = predict_n_words(prompt, n_words)
print(generated_text)
generated_text2 = predict_n_words(prompt2, n_words)
print(generated_text2)


# import gradio as gr

# demo = gr.Interface(
#     fn=predict_n_words,
#     inputs=[gr.Textbox(lines=2, placeholder="Prompt text here..."), gr.Slider(0, 500)], 
#     outputs="text",
#     examples=[
#         ["the girl from ipanema", 200],
#         ["A walk through Cagliari", 200],
#     ],
#     title="Amy Winehouse RNN",
#     description="Simple word-based RNN text generator trained on Amy Winehouse's songs",
# )

# demo.launch()