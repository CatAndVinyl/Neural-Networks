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

# makemore

**makemore** is a text generation tool built in PyTorch that takes a single text file as input, where each line represents a training example. The tool generates more text similar to the provided input using various neural network architectures, ranging from simple bi-gram models to the Wavenet architecture. These notebooks are based off Andrej Karpathy's makemore series, which consists of 5 separate videos in the series designed for learning purposes.

## Overview

The "makemore" repository contains several Jupyter Notebooks (`Makemore1.ipynb`, `Makemore2.ipynb`, etc.) each demonstrating various neural-network text generation models, best described as as a character-level autoregressive language models.

## How It Works

1. **Input Data:**
   - Prepare a text file with each line representing a training example. For instance, you can provide a list of names, company names, valid Scrabble words, etc.

2. **Jupyter Notebooks:**
   - Explore and run the specific Jupyter Notebook that corresponds to your use case (e.g., `generate_names.ipynb`).

3. **Training:**
   - Train the text generation model on your input data using a variety of available architectures.

4. **Generation:**
   - Generate more text based on the trained model. For names, this could result in unique and novel name suggestions, maintaining the style of the input.

## Example Usage

### Generate Names
1. Edit `names.txt` or replace it with another file of words
2. Open the Jupyter Notebook `Makemore5.ipynb`:

    ```bash
    jupyter notebook Makemore5.ipynb
    ```
3. Run each cell to train the model; the generated text will appear in the last cell


## Contributing

Contributions are welcome! If you have ideas for improvements, new features, or additional use cases, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
