import os
from train_tokenizer import train_tokenizer


def build_tokenizer(domain_file: str, output_dir: str, vocab_size: int, final_name: str):
    """
    Train a tokenizer with given train_tokenizer func
    """
    train_tokenizer(domain_file, output_dir, vocab_size)

    # The tokenizer is saved as "tokenizer.pkl" by default,
    # Rename it to a per-domain name (e.g., "tokenizer_1.pkl").
    original = os.path.join(output_dir, "tokenizer.pkl")
    renamed = os.path.join(output_dir, final_name)
    os.rename(original, renamed)


if __name__ == "__main__":
    domain1_path = "../data/domain_1_train.txt"
    domain2_path = "../data/domain_2_train.txt"
    domain3_path = "../data/domain_3_train.txt"
    output_dir = "../trained_tokenizer"
    vocab_size1 = 3500
    vocab_size2 = 2500
    vocab_size3 = 3500

    # Train tokenizer_1 on domain_1_train
    build_tokenizer(domain1_path, output_dir, vocab_size1, "tokenizer_1.pkl")

    # Train tokenizer_2 on domain_2_train
    build_tokenizer(domain2_path, output_dir, vocab_size2, "tokenizer_2.pkl")

    # Merge domain_1_train + domain_2_train to domain_3_train
    with open(domain3_path, 'w', encoding='utf-8') as f_out, \
            open(domain1_path, 'r', encoding='utf-8') as f1, \
            open(domain2_path, 'r', encoding='utf-8') as f2:
        f_out.writelines(f1.readlines() + f2.readlines())

    # Train tokenizer_3 on domain_3_train
    build_tokenizer(domain3_path, output_dir, vocab_size3, "tokenizer_3.pkl")
