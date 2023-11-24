# Neural-Networks

# nano-gpt

**nano-gpt** is a lightweight implementation of a simple transformer model designed to generate Shakespearean text. This project is a fun exploration of natural language processing and transformer architecture using a Jupyter Notebook. Feel free to use, modify, and experiment with the code to create your own language generation models.

## Overview

In the "nano-gpt" folder, you'll find a Jupyter Notebook (`nano_gpt.ipynb`) containing a simple implementation of the transformer architecture. The model is trained on a dataset of Shakespearean text to generate similar-sounding language. This project is educational and intended for those interested in understanding the fundamentals of transformers and natural language processing.

## Usage

1. Inside the nanogpt folder, open the Jupyter Notebook `nano-gpt.ipynb`:

   ```bash
   jupyter notebook nano-gpt.ipynb
2. Optional: edit the `input.txt` file, which contains any text you would like to create similar text from. Currently, it is a culmination of Shakespeare's works into a 1MB file.
3. Adjust the hyperparameters in the third cell containing:
   ```python
   batch_size = 64
   block_size = 32
   n_embeddings = 384
   n_heads = 6
   dropout = 0.2
4. Run through each cell including the last cell which trains the model. Then run the previous line which will begin generating text:
   ```python
   print([decode(i) for i in zed.tolist()][0])
