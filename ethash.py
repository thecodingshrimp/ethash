import copy
import os
from typing import Callable, List, Mapping
from numpy import uint32, uint64
from Crypto.Hash import keccak

uint32_hash = List[uint32]
uint32_hash_array = List[uint32_hash]

WORD_BYTES = 4                    # bytes in word
DATASET_BYTES_INIT = 2**30        # bytes in dataset at genesis
DATASET_BYTES_GROWTH = 2**23      # dataset growth per epoch
CACHE_BYTES_INIT = 2**24          # bytes in cache at genesis
CACHE_BYTES_GROWTH = 2**17        # cache growth per epoch
CACHE_MULTIPLIER=1024             # Size of the DAG relative to the cache
EPOCH_LENGTH = 30000              # blocks per epoch
MIX_BYTES = 128                   # width of mix
HASH_BYTES = 64                   # hash length in bytes
DATASET_PARENTS = 256             # number of parents of each dataset element
CACHE_ROUNDS = 3                  # number of rounds in cache production
ACCESSES = 64                     # number of accesses in hashimoto loop
FNV_PRIME = 0x01000193
REVISION = 23

def file_handler(is_cache: bool) -> Callable[[Callable[[int, str], uint32_hash_array]], Callable[[int, str], uint32_hash_array]]:
    def handling_creation(create_func: Callable[[int, str], uint32_hash_array]) -> Callable[[int, str], uint32_hash_array]:
        def get_or_create(block_nr: int, dir_path: str) -> uint32_hash_array:
            file_path = create_file_path(is_cache, dir_path, block_nr)
            if not os.path.exists(file_path):
                # file does not exist
                # create
                data = create_func(block_nr, dir_path)
                # write to file
                write_to_file(file_path, data)
                return data
            return read_from_file(file_path)
        return get_or_create
    return handling_creation

@file_handler(True)
def create_cache(block_nr: int, dir_path: str) -> uint32_hash_array:
    """Creates ethash cache and writes to dir_path if provided.

    Args:
        block_nr (int): block nr for which to create cache
        dir_path (str, optional): relative path to data dir. Defaults to None.

    Returns:
        list: uint32_hash_array
    """
    # get cache size and seed hash for the current epoch
    n = get_cache_size(block_nr) // HASH_BYTES
    seed = get_seedhash(block_nr)

    # Sequentially produce the initial dataset
    o = [keccak_512(seed)]
    for _ in range(1, n):
        o.append(keccak_512(o[-1]))

    # Use a low-round version of randmemohash
    for _ in range(CACHE_ROUNDS):
        for i in range(n):
            v = o[i][0] % n
            o[i] = keccak_512(list(map(xor, o[(i-1+n) % n], o[v])))

    return o

@file_handler(False)    
def create_dataset(block_nr: int, dir_path: str) -> uint32_hash_array:
    """
        Creates ethash dataset and writes to provided dir_path.

        Returns a list of type uint32_hash_array.

    Args:
        block_nr (int): block nr for which to create cache
        dir_path (str): relative path to data dir. Defaults to None.

    Returns:
        list: uint32_hash_array
    """
    cache: uint32_hash_array = create_cache(block_nr, dir_path)
    full_size: uint32 = get_full_size(block_nr)
    dataset: uint32_hash_array = [calc_dataset_item(cache, uint32(i)) for i in range(full_size // HASH_BYTES)]
    
    return dataset

def hashimoto(dataset_lookup: Callable[[uint32_hash_array, uint32], uint32_hash]) -> Callable[[bytearray, bytearray, int, uint32_hash_array], Mapping[str, bytes]]:
    """ Decorator function. Returns the hashimoto algorithm.

    Args:
        dataset_lookup (Callable[[list, int], List[int]]): data item lookup function

    Returns:
        Callable[[bytearray, bytearray, int], Dict[str, str]]: hashimoto algorithm function
    """
    def algorithm(header: bytearray, nonce: bytearray, block_nr: int, dataset: uint32_hash_array) -> Mapping[str, bytes]:
        full_size = get_full_size(block_nr)
        n = full_size // HASH_BYTES
        w = MIX_BYTES // WORD_BYTES
        mixhashes = MIX_BYTES / HASH_BYTES
        # combine header+nonce into a 64 byte seed
        header += nonce[::-1]
        s: uint32_hash = keccak_512(header)
        # start the mix with replicated s
        mix: List[uint32] = []
        for _ in range(MIX_BYTES // HASH_BYTES):
            mix.extend(s)
        # mix in random dataset nodes
        for i in range(ACCESSES):
            p = fnv(i ^ s[0], mix[i % w]) % (n // mixhashes) * mixhashes
            newdata = []
            for j in range(MIX_BYTES // HASH_BYTES):
                newdata.extend(dataset_lookup(dataset, uint32(p + j)))
            mix = list(map(fnv, mix, newdata))
        # compress mix
        cmix: List[uint32] = []
        for i in range(0, len(mix), 4):
            cmix.append(fnv(fnv(fnv(mix[i], mix[i+1]), mix[i+2]), mix[i+3]))
        return {
            "mix digest": serialize_hash(cmix),
            "result": serialize_hash(keccak_256(s+cmix))
        }

    return algorithm

@hashimoto
def hashimoto_light(cache: uint32_hash_array, index: uint32) -> uint32_hash:
    """Computes dataset item from cache.

    Args:
        cache (uint32_hash_array): full size cache in memory
        index (int): index

    Returns:
        List[int]: uint_32[16]
    """
    return calc_dataset_item(cache, index)

@hashimoto
def hashimoto_full(dataset: uint32_hash_array, x: uint32) -> uint32_hash:
    """Gets dataset item from dataset.

    Args:
        dataset (uint32_hash_array): full size dataset in memory
        index (int): index

    Returns:
        List[int]: uint_32[16]
    """
    return dataset[x]

def calc_dataset_item(cache: uint32_hash_array, i: uint32) -> uint32_hash:
    n = len(cache)
    r = HASH_BYTES // WORD_BYTES
    # initialize the mix
    mix = copy.copy(cache[i % n])
    mix[0] ^= i
    mix = keccak_512(mix)
    # fnv it with a lot of random cache nodes based on i
    for j in range(DATASET_PARENTS):
        cache_index = fnv(i ^ j, mix[j % r])
        mix = list(map(fnv, mix, cache[cache_index % n]))
    result = keccak_512(mix)
    return result

def get_cache_size(block_number: int) -> int:
    sz = CACHE_BYTES_INIT + CACHE_BYTES_GROWTH * (block_number // EPOCH_LENGTH)
    sz -= HASH_BYTES
    while not isprime(sz / HASH_BYTES):
        sz -= 2 * HASH_BYTES
    return sz

def get_full_size(block_number: int) -> uint32:
    sz = DATASET_BYTES_INIT + DATASET_BYTES_GROWTH * (block_number // EPOCH_LENGTH)
    sz -= MIX_BYTES
    while not isprime(sz / MIX_BYTES):
        sz -= 2 * MIX_BYTES
    return uint32(sz)

def fnv(v1: uint64, v2: uint64) -> uint32:
    return uint32(((v1 * FNV_PRIME) ^ v2) % 2**32)

def get_seedhash(block_number: int) -> bytes:
    s = b'\x00' * 32
    for _ in range(block_number // EPOCH_LENGTH):
        s = serialize_hash(keccak_256(s))
    return s

def create_file_path(is_cache: bool, dir: str, block_nr: int) -> str:        
    dir = f'{dir}/' if dir[-1] != '/' else dir
    if not os.path.isdir(dir):
        os.mkdir(dir)
    file_path = f'{dir}{"cache" if is_cache else "dataset"}_rev_{REVISION}_epoch_{block_nr // EPOCH_LENGTH}'
    return file_path

# Assumes little endian bit ordering (same as Intel architectures)
def decode_int(s: bytes) -> uint32:
    """Decodes little endian encoded hex byte to little endian int

    Args:
        s (bytes): big endian encoded int

    Returns:
        int: little endian int
    """
    return uint32(int(s[::-1].hex(), 16)) if s else 0

def encode_int(s: uint32) -> bytes:
    """Encodes int to little endian hex

    Args:
        s (uint32): uint32 to encode

    Returns:
        bytes: encoded uint32 in little endian by
    """
    return s.tobytes()

def serialize_hash(h: uint32_hash) -> bytes:
    """ Serialize hash

    Args:
        h (list): hash

    Returns:
        str: serialized hashed
    """
    new_array = [encode_int(x) for x in h]
    try:
        return b''.join(new_array)
    except:
        print('nope serialize hash')
        exit(-1)

def deserialize_hash(h: bytes) -> uint32_hash:
    return [decode_int(h[i:i+WORD_BYTES]) for i in range(0, len(h), WORD_BYTES)]

def hash_words(h, x) -> uint32_hash:
    if isinstance(x, list):
        x = serialize_hash(x)
    y = h(x)
    return deserialize_hash(y)

def serialize_cache(ds):
    return ''.join([serialize_hash(h) for h in ds])

def write_to_file(file_path: str, data: uint32_hash_array):
    """Takes uint32_hash_array and writes every entry as hex to file.

    Args:
        file_path (str): relative file path
        data (uint32_hash_array): uint_32[*][16]
    """
    with open(file_path, 'w') as f:
        for num_array in data:
            concatenated_hex = ''.join([int(num).to_bytes(4, 'big').hex() for num in num_array])
            f.write(f'{concatenated_hex}\n')
        
def read_from_file(file_path: str) -> uint32_hash_array:
    """Reads 64byte hex hashes into uint_32[*][16] array

    Args:
        file_path (str): relative path to file

    Returns:
        list: uint_32[*][16] array
    """
    with open(file_path, 'r') as f:
        raw_data = f.read()
        split_data = raw_data.split('\n')
        if len(split_data) <= 10:
            print(f'path: {file_path} is empty or has unsufficient entries to be dataset/cache. Remove it before we can continue.')
            exit(-1)
        # if there is a '\n' at the end of the file, remove last empty item in list
        if len(split_data[len(split_data) - 1]) == 0:
            split_data.pop(len(split_data) - 1)
        uint32_array = [[uint32(int(x[i:i+8], 16)) for i in range(0, len(x), 8)] for x in split_data]
        return uint32_array

serialize_dataset = serialize_cache

# keccak hash function, outputs 64 byte array
def keccak_512(x) -> uint32_hash:
    return hash_words(lambda v: keccak.new(data=v, digest_bits=512).digest(), x)

def keccak_256(x: uint32_hash) -> uint32_hash:
    return hash_words(lambda v: keccak.new(data=v, digest_bits=256).digest(), x)

def xor_bytes(a, b) -> bytes:
    return bytes([_a ^ _b for _a, _b in zip(a, b)])

def xor(a, b):
    return a ^ b

def isprime(x) -> bool:
    for i in range(2, int(x**0.5)):
         if x % i == 0:
             return False
    return True

