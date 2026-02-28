import re
import torch
from tokenizer.simple_tokenizer_v1 import SimpleTokenizerV1
from tokenizer.simple_tokenizer_v2 import SimpleTokenizerV2
from gpt_dataset.gpt_dataset_v1 import GPTDatasetV1, create_dataloader_v1
import tiktoken

torch.manual_seed(123)

def read_file(text_file: str) -> str:
    with open(f"{text_file}", "r", encoding="utf-8") as f:
        raw_text = f.read()
    return raw_text


def create_vocabulary(text: str):
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    all_words = sorted(set(preprocessed))

    # Adding special characters: unk and endoftext
    all_words.extend(["<|endoftext|>", "<|unk|>"])
    return {token: integer for integer, token in enumerate(all_words)}


def main():

    # Tokenization
    raw_text = read_file("data/the-verdict.txt")

    max_length = 4

    dataloader = create_dataloader_v1(
        raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)

    print("Token IDs:\n", inputs)
    print("\nInputs shape:\n", inputs.shape)  # 8 x 4

    vocab_size = 50257
    output_dim = 256

    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

    token_embeddings = token_embedding_layer(inputs)
    print(token_embeddings.shape)  # 8 x 4 x 256

    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    print(pos_embeddings.shape)  # 4 x 256

    # Now we can add the positional embedding to the token embeddings
    input_embeddings = token_embeddings + pos_embeddings
    print(input_embeddings.shape)


if __name__ == "__main__":
    main()
