import re
from tokenizer.simple_tokenizer_v1 import SimpleTokenizerV1


def read_file(text_file: str):
    with open(f"{text_file}", "r", encoding="utf-8") as f:
        raw_text = f.read()
    return raw_text


def main():

    raw_text = read_file("data/the-verdict.txt")

    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    all_words = sorted(set(preprocessed))
    vocab = {token: integer for integer, token in enumerate(all_words)}

    tokenizer = SimpleTokenizerV1(vocab)

    text = """It's the last he painted, you know,
            Mrs.Gisburn said with pardonable pride."""

    ids = tokenizer.encode(text)
    print(ids)

    print(tokenizer.decode(ids))


if __name__ == "__main__":
    main()
