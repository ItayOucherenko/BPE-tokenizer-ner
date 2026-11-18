from base_tokenizer import BaseTokenizer
from typing import List, Dict, Tuple
from collections import Counter, defaultdict


class BPETokenizer(BaseTokenizer):
    """
    BPE (Byte Pair Encoding) Tokenizer implementation
    """

    def __init__(self, vocab_size: int = 10000):
        """
        Initializes BPE tokenizer
        :param vocab_size: Maximum vocabulary size
        """
        super().__init__()

        # Characters are represented using UTF-8 bytes, so the space_token needs to be also in UTF-8
        self.space_token = " ".encode('utf-8')
        self.vocab_size = vocab_size

        # Structures to store merge rules
        self.merges = []        # Ordered list of merged pairs
        self.merge_dict = {}    # Maps each merged pair to its assigned token ID
        self.merge_order = {}   # Tracks the order in which each pair was merged

        # Used for fast lookup of which words contain a given pair
        self.pair_to_words = defaultdict(set)   # pair -> set of words that contain it
        self.pair_counts = Counter()            # pair -> frequency in corpus

        # Next available token ID for merged tokens
        # Starts at 256 because all UTF-8 byte values (0–255) are already assigned
        self.next_id = 256

        # Initialize vocabulary with base UTF-8 bytes (0–255)
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

        # Fill token_to_id and id_to_token with base UTF-8 bytes (0–255)
        # skip special tokens - IDs 0–3, so add 4 to id
        for i in range(256):
            token = bytes([i])
            self.token_to_id[token] = i + 4
            self.id_to_token[i + 4] = token

        # Assigns IDs to space+byte UTF-8 tokens to mark first character of a word
        for i in range(256):
            token = self.space_token + bytes([i])
            self.vocab[self.next_id] = token    # Don't contain special tokens, so no need for +4
            self.token_to_id[token] = self.next_id + 4
            self.id_to_token[self.next_id + 4] = token
            self.next_id += 1

    def _get_word_freqs(self, texts: List[str]) -> Dict[Tuple[int, ...], int]:
        """
        Converts input to UTF-8 byte words sequences and computes the frequency
        of each byte pair of the words across the corpus.
        """
        word_freqs = Counter()

        for text in texts:
            if not text:
                continue

            text = text.strip().split()
            for word in text:
                word_bytes = word.encode('utf-8')

                # Change ID to first byte to space_token + first byte to indicate word start
                first_token = self.token_to_id[self.space_token + bytes([word_bytes[0]])] - 4
                rest = list(word_bytes[1:])

                word_freqs[tuple([first_token] + rest)] += 1

        return word_freqs

    def _build_pair_index(self, word_freqs: Dict[Tuple[int, ...], int]) -> None:
        """
        Builds an index that maps each token pair to:
        - set of words containing that pair (self.pair_to_words)
        - total frequency of that pair in the corpus (self.pair_counts)
        """
        self.pair_to_words.clear()
        self.pair_counts.clear()

        for word, freq in word_freqs.items():
            if len(word) < 2:
                continue

            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                self.pair_to_words[pair].add(word)
                self.pair_counts[pair] += freq

    def _update_pair_index_remove_word(self, word: Tuple[int, ...], freq: int) -> None:
        """
        Updates the index by removing all pairs found in the given word, and reducing their frequency accordingly.
        Used before replacing the word with its merged form.
        """
        if len(word) < 2:
            return

        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])

            # Remove the word from the set of words containing this pair
            self.pair_to_words[pair].discard(word)

            # Decrease the pair's overall frequency count
            self.pair_counts[pair] -= freq

            # Clean up if the pair no longer exists in any word
            if self.pair_counts[pair] <= 0:
                del self.pair_counts[pair]
                if pair in self.pair_to_words:
                    del self.pair_to_words[pair]

    def _update_pair_index_add_word(self, word: Tuple[int, ...], freq: int) -> None:
        """
        Adds all byte pairs from the given word into the pair index,
        and increases their frequency by the word's frequency.
        Called after creating a new merged word.
        """
        if len(word) < 2:
            return

        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            self.pair_to_words[pair].add(word)
            self.pair_counts[pair] += freq

    def _merge_word(self, word: Tuple[int, ...], pair: Tuple[int, int], new_token_id: int) -> Tuple[int, ...]:
        """
        Called after selecting a pair to merge.
        Replaces all its appearances in the word with the new token ID.
        """
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(new_token_id)
                i += 2
            else:
                new_word.append(word[i])
                i += 1

        return tuple(new_word)

    def _find_common_and_uncommon_bigrams(self, texts: List[str]) -> None:
        """
        Finds the 5 most and least frequent word bigrams, prints them with their frequencies,
        and add 1 bigram to the roles - the most frequent one.
        """
        bigram_counter = Counter()

        # Count all word bigrams in the text
        for text in texts:
            words = text.strip().split()
            bigrams = zip(words, words[1:])
            bigram_counter.update(bigrams)

        # Get most and least frequent bigrams
        most_common_bigrams = bigram_counter.most_common(5)
        least_common_bigrams = bigram_counter.most_common()[-5:]

        print("\nTop 5 most frequent bigrams:")
        for pair, count in most_common_bigrams:
            print(f"    '{pair[0]} + {pair[1]}' appeared {count} times")

        print("\nTop 5 least frequent bigrams:")
        for pair, count in least_common_bigrams:
            print(f"    '{pair[0]} + {pair[1]}' appeared {count} times")

        # Get most frequent bigram and convert both words to UTF-8 bytes
        most_common_bigram, freq = most_common_bigrams[0]
        word1 = most_common_bigram[0].encode("utf-8")
        word2 = most_common_bigram[1].encode("utf-8")

        # Add space token to mark start of word (like in _get_word_freqs)
        word1_token = self.space_token + word1
        word2_token = self.space_token + word2

        token1 = self.token_to_id[word1_token] - 4
        token2 = self.token_to_id[word2_token] - 4

        # Assign new token ID for merged bigram
        new_token_id = self.next_id
        self.next_id += 1

        # Store merge rule
        self.merges.append((token1, token2))
        self.merge_dict[(token1, token2)] = new_token_id
        self.merge_order[(token1, token2)] = len(self.merges) - 1

        # Add to vocab
        merged_token_bytes = word1 + word2
        self.vocab[new_token_id] = merged_token_bytes
        self.token_to_id[merged_token_bytes] = new_token_id + 4
        self.id_to_token[new_token_id + 4] = merged_token_bytes
        print(f"\nIteration {len(self.merges) - 1}  -->  Merging {(token1, token2)} = '{word1.decode()} + {word2.decode()}'"
              f"  | appeared: {freq:,} times\n")

    def train(self, texts: List[str]) -> None:
        """
        Trains the BPE tokenizer by merging frequent byte pairs step by step,
        until the vocabulary reaches the specified size.
        """
        print(f"Training BPE on {len(texts)} texts...")

        # Step 1: Convert each word into UTF-8 token tuples and count their frequencies
        word_freqs = self._get_word_freqs(texts)
        print(f"Found {len(word_freqs)} unique words")

        # Step 2: Initialize pair index with all adjacent token pairs and their frequencies
        self._build_pair_index(word_freqs)
        print(f"Initial pairs: {len(self.pair_counts)}")

        # Ensure it's a regular dict for better performance
        self.vocab = dict(self.vocab)

        # Step 3: Perform merges until reaching target vocab size
        num_merges = self.vocab_size - len(self.vocab)
        for i in range(num_merges):
            if not self.pair_counts:
                break   # No more pairs to merge

            # Select the most frequent token pair in the corpus
            best_pair = max(self.pair_counts.items(), key=lambda x: x[1])[0]
            best_pair_freq = self.pair_counts[best_pair]

            # Assign a new token ID for the merged pair
            new_token_id = self.next_id
            self.next_id += 1

            # Store merge info before processing (avoid redundant lookups)
            self.merges.append(best_pair)
            self.merge_dict[best_pair] = new_token_id
            self.merge_order[best_pair] = i     # Used later in encoding

            # Pre-compute new token bytes
            new_token_bytes = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.vocab[new_token_id] = new_token_bytes
            self.token_to_id[new_token_bytes] = new_token_id + 4
            self.id_to_token[new_token_id + 4] = new_token_bytes

            left = self.vocab[best_pair[0]]
            right = self.vocab[best_pair[1]]
            merged_str = f"{left.decode(errors='replace')} + {right.decode(errors='replace')}"
            print(f"Iteration {i}  -->  Merging {best_pair} = '{merged_str}'  | appeared: {best_pair_freq:,} times")

            # Get all words that contain the selected pair
            words_with_pair = list(self.pair_to_words.get(best_pair, set()))

            # Filter only those still in word_freqs (i.e., still active)
            valid_words_and_freqs = [(word, word_freqs[word]) for word in words_with_pair if word in word_freqs]

            if not valid_words_and_freqs:
                continue

            # Remove the old versions of these words from the pair index
            for word, freq in valid_words_and_freqs:
                self._update_pair_index_remove_word(word, freq)

            # Apply the merge to all valid words and collect new forms
            new_words_and_freqs = []
            for word, freq in valid_words_and_freqs:
                new_word_tuple = self._merge_word(word, best_pair, new_token_id)
                new_words_and_freqs.append((new_word_tuple, freq))

            # Replace old word forms with the new ones in the word frequency dictionary
            for word, freq in valid_words_and_freqs:
                del word_freqs[word]

            for new_word, freq in new_words_and_freqs:
                word_freqs[new_word] = freq

            # Add the new merged words back to the pair index
            for new_word, freq in new_words_and_freqs:
                self._update_pair_index_add_word(new_word, freq)

        self._find_common_and_uncommon_bigrams(texts)

    def _apply_merges(self, tokens: List[int]) -> List[int]:
        """
        Applies BPE merges to a token list by selecting the highest-priority pair
        (earliest merge index). Used only in encode func.
        """
        if len(tokens) < 2:
            return tokens   # Nothing to merge

        while True:
            best_pos = None
            best_pair = None
            best_order = float('inf')   # Lower is higher priority (earlier merge)

            # Find the pair with the earliest merge order
            for i, (a, b) in enumerate(zip(tokens, tokens[1:])):
                pair = (a, b)
                order = self.merge_order.get(pair)
                if order is not None and order < best_order:
                    best_order = order
                    best_pair = pair
                    best_pos = i

            if best_pair is None:
                break   # No more mergeable pairs

            # Merge the best pair in-place
            merged_token = self.merge_dict[best_pair]
            tokens[best_pos:best_pos + 2] = [merged_token]

        return tokens

    def encode(self, text: str) -> List[int]:
        """
        Encode text by applying merges to each word separately.
        Preserves UTF-8 words and replaces each word’s first byte with space+byte ID.
        """
        if not text:
            return []

        words = text.encode('utf-8').split(b' ')
        token_ids = []

        for word in words:
            if not word:
                continue    # Skip empty words (e.g., multiple spaces)

            # Change first byte to space_token + first byte ID to mark word start
            first_token = self.space_token + bytes([word[0]])
            word_bytes = [self.token_to_id[first_token] - 4]

            # Add the remaining bytes as-is
            word_bytes += list(word[1:])
            merged = self._apply_merges(word_bytes)
            token_ids.extend(merged)

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Converts vocab token IDs to UTF-8 bytes and decodes them back to text.
        """
        if not token_ids:
            return ""

        # Convert each token ID back to its byte representation and join them
        byte_sequence = b''.join(self.vocab[token_id] for token_id in token_ids if token_id in self.vocab)

        # Decode UTF-8 bytes into text
        return byte_sequence.decode('utf-8', errors='replace')
