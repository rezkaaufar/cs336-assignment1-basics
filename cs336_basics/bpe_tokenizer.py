import pickle
import regex as re
from collections.abc import Iterable, Iterator
from collections import OrderedDict
import heapq

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(
        self, 
        vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]], 
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.merge_dictionary = {}
        self.pat = re.compile(PAT)

        for i, key in enumerate(merges):
            self.merge_dictionary[key] = i

        self.inverted_vocab = {}
        for key in self.vocab:
            self.inverted_vocab[self.vocab[key]] = key

    @classmethod
    def from_files(
        cls, 
        vocab_filepath: str, 
        merges_filepath: str, 
        special_tokens: list[str] | None = None
    ):
        with open(vocab_filepath, "rb") as file:
            vocab = pickle.load(file)
        with open(merges_filepath, "rb") as file:
            merges = pickle.load(file)
        
        return cls(vocab, merges, special_tokens)
        

    def pretokenize(self, text):
        if not self.special_tokens:  # handles None or empty list/tuple
            return self.pat.findall(text)
        # sort longest-first so overlapping special tokens don't shadow each other
        specials = sorted(self.special_tokens, key=len, reverse=True)
        split_pat = "(" + "|".join(re.escape(t) for t in specials) + ")"

        pieces = re.split(split_pat, text)  # capturing group keeps the delimiters
        out = []
        for piece in pieces:
            if piece in self.special_tokens:
                out.append(piece)          # keep special token whole
            elif piece:
                out.extend(self.pat.findall(piece))  # normal tokenization
        return out


    def encode(self, text: str) -> list[int]:
        # texts = self.pat.findall(text)
        texts = self.pretokenize(text)
        # print(texts)
        result = []
        set_st = set()
        if self.special_tokens:
            set_st = set(self.special_tokens)
        for token in texts:
            byte_list = [bytes([b]) for b in token.encode("utf-8")]
            if self.special_tokens and token in set_st:
                result.append(token.encode("utf-8"))
                continue

            next_idx = [i+1 for i in range(len(byte_list))]
            prev_idx = [i-1 for i in range(len(byte_list))]
            next_idx[-1] = -1
            deleted = [False] * len(byte_list)

            # create min heap pairs
            pair_priority = []
            heapq.heapify(pair_priority)
            for i, p in enumerate(zip(byte_list, byte_list[1:])):
                if p in self.merge_dictionary:
                    rank = self.merge_dictionary[p]
                    heapq.heappush(pair_priority, (rank, i))


            while pair_priority:
                rank, left_idx = heapq.heappop(pair_priority)
                # print(pair_priority, rank, left_idx)

                # print(byte_list, deleted, pair_priority)

                if deleted[left_idx]:
                    continue
                right_idx = next_idx[left_idx]
                if right_idx == -1:
                    continue
                # this is to check stale case if (A,B,C) B,C is rank 1 and A,B is rank 2, if B,C gets merged then A,B is not valid anymore
                if self.merge_dictionary.get((byte_list[left_idx], byte_list[right_idx])) != rank:
                    continue

                # merge right into left in O(1)
                byte_list[left_idx] = byte_list[left_idx] + byte_list[right_idx]
                deleted[right_idx] = True

                # update next idx for left
                next_idx[left_idx] = next_idx[right_idx]
                # update prev idx for new right
                if next_idx[right_idx] != -1:
                    prev_idx[next_idx[right_idx]] = left_idx

                # Check newly formed neighbor pairs and push to heap
                # Left neighbor: (byte_list[prev], byte_list[left_idx])
                left_neighbor = prev_idx[left_idx]
                if left_neighbor != -1:
                    left_pair = (byte_list[left_neighbor], byte_list[left_idx])
                    # print(left_pair)
                    if left_pair in self.merge_dictionary:
                        rank = self.merge_dictionary[left_pair]
                        heapq.heappush(pair_priority, (rank, left_neighbor))

                # Right neighbor: (byte_list[left], byte_list[next])
                right_neighbor = next_idx[left_idx]
                if right_neighbor != -1:
                    right_pair = (byte_list[left_idx], byte_list[right_neighbor])
                    # print(right_pair)
                    if right_pair in self.merge_dictionary:
                        rank = self.merge_dictionary[right_pair]
                        heapq.heappush(pair_priority, (rank, left_idx))

            for res, stat in zip(byte_list, deleted):
                if not stat:
                    result.append(res)
        # print(result)
        
        return [self.inverted_vocab[elem] for elem in result]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for lines in iterable:
            for ch_int in self.encode(lines):
                yield ch_int
        
    def decode(self, ids: list[int]) -> str:
        bytes_res = b''
        for id in ids:
            bytes_res  += self.vocab[id]
        # print(bytes_res)
        return bytes_res.decode("utf-8", errors='replace')


if __name__ == '__main__':
    vocab_path = "data/vocab_tinystories.pkl"
    merges_path = "data/merges_tinystories.pkl"

    tokenizer = Tokenizer.from_files(vocab_path, merges_path, ["<|endoftext|>",])

    # text_input = "Hello, how are you?"
    # text_input = "Héllò hôw are ü? 🙃"
    text_input = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"

    # ids = tokenizer.encode("my son and my wife are the best person ever in the world")
    # ids = tokenizer.encode("🙃")
    ids = tokenizer.encode(text_input)
    # print(ids)
    tokenized_string = [tokenizer.decode([x]) for x in ids]
    # print(tokenized_string)
    # print(tokenizer.decode(ids))
