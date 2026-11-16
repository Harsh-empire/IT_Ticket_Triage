from transformer_training import train_and_evaluate


def main():
    # Keep the same split, sequence length, batch size, and epoch count.
    train_and_evaluate(
        "distilbert-base-multilingual-cased",
        "distilbert",
    )
    train_and_evaluate(
        "bert-base-multilingual-cased",
        "bert",
    )


if __name__ == "__main__":
    main()
