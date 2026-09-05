import os
from typing import BinaryIO
import multiprocessing as mp
from collections import Counter

from pathlib import Path
DATA_PATH = Path(__file__).parent.parent / "data" / "TinyStoriesV2-GPT4-valid.txt"

from cs336_basics.bpe_training import build_counts

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

def parallel_word_count(num_processes: int) -> dict[tuple[str], int]:

    with open(DATA_PATH, "rb") as f:
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
        
        print("Top 3 Consolidated Word Counts:")
        print(final_counts.most_common(3))

        return final_counts

## Usage
if __name__ == '__main__':
    fc = parallel_word_count(num_processes=100)