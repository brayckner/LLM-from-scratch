import re
from tokenizer.simple_tokenizer_v1 import SimpleTokenizerV1
from tokenizer.simple_tokenizer_v2 import SimpleTokenizerV2
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

    raw_text = read_file("data/the-verdict.txt")

    vocab = create_vocabulary(raw_text)

    tokenizer = tiktoken.get_encoding("gpt2")

    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = " <|endoftext|> ".join((text1, text2))
    print(text)

    ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print(ids)

    print(tokenizer.decode(ids))


if __name__ == "__main__":
    main()
