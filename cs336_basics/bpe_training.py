from collections import Counter, defaultdict
import regex as re

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

    ### char version ###
    init_d = Counter(splitted_chunk)
    fin_d = {}
    for key in init_d:
        new_key = tuple(list(key))
        fin_d[new_key] = init_d[key]

    return fin_d

    ### byte version ###
    # byte_list = []
    # for word in splitted_chunk:
    #     byte_key = word.encode("utf-8")
    #     byte_list.append(byte_key)
    # return [list(w) for w in Counter(byte_list)]

# def train_bpe(
#     file_path: str, 
#     vocab_size: int, 
#     special_tokens: bytes = b"<|endoftext|>"
# ) -> (dict[int, bytes], list[tuple[bytes, bytes]]):

#     # update pairs and pos
#     pairs = defaultdict(int)
#     pos = defaultdict(set)
#     idx = 0
#     for word, freq in zip(words, counts):
#         for p in zip(word, word[1:]):
#         pairs[p] += freq
#         pos[p].add(idx)
#         idx += 1

#     for _ in range(6):
#     best = max(pairs, key=lambda p: (pairs[p], p))
#     # merging with caching
#     eligible_words_idx = pos[best]
#     for idx in list(eligible_words_idx):
#         word = words[idx]
#         c = counts[idx]
#         a, b = best
#         out, i, n = [], 0, len(word)
#         while i < n:
#         if i < n - 1 and word[i] == a and word[i + 1] == b:
#             out.append(a + b)
#             i += 2          # skip both — no overlap
#         else:
#             out.append(word[i])
#             i += 1
#         # update pairs and pos to 0
#         for p in zip(word, word[1:]):
#         pairs[p] = 0
#         pos[p].add(idx)

#         # update pairs and pos using the new merge
#         for p in zip(out, out[1:]):
#         pairs[p] += c
#         pos[p].add(idx)

#         # remove merged pairs and pos
#         pairs.pop(best, None)
#         pos.pop(best, None)

#         # update new words
#         words[idx] = out