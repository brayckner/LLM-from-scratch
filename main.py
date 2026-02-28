import re
from tokenizer.simple_tokenizer_v1 import SimpleTokenizerV1
from tokenizer.simple_tokenizer_v2 import SimpleTokenizerV2
from gpt_dataset.gpt_dataset_v1 import GPTDatasetV1, create_dataloader_v1
import tiktoken


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

    # tokenizer = tiktoken.get_encoding("gpt2")  # Utilizing BPE tokenizer, same as GPT-2

    # enc_text = tokenizer.encode(raw_text)
    # print(len(enc_text))

    # # Tokens to input-target pairs
    # enc_sample = enc_text[50:]

    # context_size = 4
    # x = enc_sample[:context_size]
    # y = enc_sample[1 : context_size + 1]

    # print(f"x: {x}")  # Inputs
    # print(f"y:      {y}")  # Targets

    # # Next word prediction task (using tokenIDs)
    # for i in range(1, context_size + 1):
    #     context = enc_sample[:i]
    #     desired = enc_sample[i]
    #     print(context, "----->", desired)

    # # Next word prediction task (using words for visual)
    # for i in range(1, context_size + 1):
    #     context = enc_sample[:i]
    #     desired = enc_sample[i]
    #     print(tokenizer.decode(context), "----->", tokenizer.decode([desired]))

    # Batch Size 1
    dataloader = create_dataloader_v1(
        raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
    )
    data_iter = iter(dataloader)

    first_batch = next(data_iter)
    print(first_batch)

    second_batch = next(data_iter)
    print(second_batch)

    # Batch Size 8
    dataloader = create_dataloader_v1(
        raw_text, batch_size=8, max_length=4, stride=4, shuffle=False
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)

    print("Inputs:\n", inputs)
    print("\nTargets:\n", targets)


if __name__ == "__main__":
    main()
