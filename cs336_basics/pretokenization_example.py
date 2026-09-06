import os
from typing import BinaryIO
import multiprocessing as mp
from collections import Counter, defaultdict
import regex as re

from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "TinyStoriesV2-GPT4-valid.txt"
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

"""
# initialize vocab list on byte level

special_token = "<|endoftext|>"
vocab_list = [bytes([idx]) for idx in range(256)]
vocab_list.append(special_token.encode("utf-8"))
"""

def build_counts(
    chunk: str
):
    splitted_chunk = re.findall(PAT, chunk)

    # ### char version ###
    # init_d = Counter(splitted_chunk)
    # fin_d = {}
    # for key in init_d:
    #     new_key = tuple(list(key))
    #     fin_d[new_key] = init_d[key]

    ### byte version ###
    init_d = Counter(splitted_chunk)
    fin_d = {}
    for key in init_d:
        byte_list = [ch.encode("utf-8") for ch in key]
        byte_key = tuple(byte_list)
        fin_d[byte_key] = init_d[key]
    
    return fin_d

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def parallel_word_count(
    file_path: str,
    num_processes: int
) -> dict[tuple[str], int]:

    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        doc_chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            splitted_chunks = chunk.split("<|endoftext|>")
            for elem in splitted_chunks:
                doc_chunks.append(elem)

        num_cores = mp.cpu_count()

        # returns every single dict tuple bytes count from each chunk
        with mp.Pool(processes=num_cores) as pool:
            results = pool.map(build_counts, doc_chunks)

        # consolidate all the result in one final dictionary
        final_counts = Counter()

        # Loop through the results and merge them into the master counter
        for chunk_counter in results:
            final_counts.update(chunk_counter)
        
        # print("Top 3 Consolidated Word Counts:")
        # print(final_counts.most_common(3))

        return final_counts

def train_bpe(
    file_path: str, 
    vocab_size: int, 
    special_tokens: list[str] = ["<|endoftext|>"]
) -> (dict[int, bytes], list[tuple[bytes, bytes]]):

    # initialize initial vocab
    special_token_id = 0
    vocab = {}
    vocab[special_token_id] = special_tokens[0].encode("utf-8")
    for idx in range(256):
        vocab[idx+1] = bytes([idx]) 
    token_id = idx + 2

    # initialize merges
    merges = []

    # calculate parallel pretokenization
    fc = parallel_word_count(file_path=file_path, num_processes=100)

    # build unique words and unique counts
    words = [w for w in fc.keys()]
    counts = [c for k, c in fc.items()]

    # update pairs and pos
    pairs = defaultdict(int)
    pos = defaultdict(set)
    idx = 0
    for word, freq in zip(words, counts):
        for p in zip(word, word[1:]):
            pairs[p] += freq
            pos[p].add(idx)
        idx += 1

    while token_id < vocab_size:
        best = max(pairs, key=lambda p: (pairs[p], p))
        # print(best)
        # print(dict(sorted(pairs.items(), key=lambda item: item[1], reverse=True)))
        # print()
        
        # build merged vocab
        p1, p2 = best
        merged_ch = p1 + p2
        # merged_tuple = (p1.encode("utf-8"), p2.encode("utf-8"))
        # merges.append(merged_tuple)
        # vocab[token_id] = merged_ch.encode("utf-8")

        merged_tuple = (p1, p2)
        merges.append(merged_tuple)
        vocab[token_id] = merged_ch
        token_id += 1

        eligible_words_idx = pos[best]
        for idx in list(eligible_words_idx):
            # print(idx, len(words))
            word = words[idx]
            c = counts[idx]
            out, i, n = [], 0, len(word)
            while i < n:
                # merging happen
                if i < n - 1 and word[i] == p1 and word[i + 1] == p2:
                    out.append(merged_ch)
                    i += 2          # skip both — no overlap
                else:
                    out.append(word[i])
                    i += 1
            # update pairs to 0
            for p in zip(word, word[1:]):
                pairs[p] -= c

            # update pairs and pos using the new merge
            for p in zip(out, out[1:]):
                pairs[p] += c
                pos[p].add(idx)

            # remove merged pairs and pos
            pairs.pop(best, None)
            pos.pop(best, None)

            # update new words
            words[idx] = out
    
    return vocab, merges


DATA_PATH = Path(__file__).parent.parent / "tests" / "fixtures" / "corpus.en"
## Usage
if __name__ == '__main__':
    vocab, merges = train_bpe(DATA_PATH, 500)
    # print(vocab)
    print(merges)