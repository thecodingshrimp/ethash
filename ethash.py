import copy
import os
from typing import Callable, List, Mapping
from numpy import uint64
from Crypto.Hash import keccak

uint32_hash = List[int]
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
                data = create_func(block_nr, file_path)
                # write to file
                write_to_file(file_path, data)
                return data
            element_amount = (get_cache_size(block_nr) if is_cache else get_full_size(block_nr)) // HASH_BYTES
            return read_from_file(file_path, element_amount)
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
    full_size: int = get_full_size(block_nr)
    dataset: uint32_hash_array = [calc_dataset_item(cache, i) for i in range(full_size // HASH_BYTES)]
    
    return dataset

def hashimoto(dataset_lookup: Callable[[uint32_hash_array, int], uint32_hash]) -> Callable[[bytearray, bytearray, int, uint32_hash_array], Mapping[str, bytes]]:
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
        mix: List[int] = []
        for _ in range(MIX_BYTES // HASH_BYTES):
            mix.extend(s)
        # mix in random dataset nodes
        for i in range(ACCESSES):
            p = fnv(i ^ s[0], mix[i % w]) % (n // mixhashes) * mixhashes
            newdata = []
            for j in range(MIX_BYTES // HASH_BYTES):
                newdata.extend(dataset_lookup(dataset, int(p + j)))
            mix = list(map(fnv, mix, newdata))
        # compress mix
        cmix: List[int] = []
        for i in range(0, len(mix), 4):
            cmix.append(fnv(fnv(fnv(mix[i], mix[i+1]), mix[i+2]), mix[i+3]))
        return {
            "mix digest": serialize_hash(cmix),
            "result": serialize_hash(keccak_256(s+cmix))
        }

    return algorithm

@hashimoto
def hashimoto_light(cache: uint32_hash_array, index: int) -> uint32_hash:
    """Computes dataset item from cache.

    Args:
        cache (uint32_hash_array): full size cache in memory
        index (int): index

    Returns:
        List[int]: uint_32[16]
    """
    return calc_dataset_item(cache, index)

@hashimoto
def hashimoto_full(dataset: uint32_hash_array, x: int) -> uint32_hash:
    """Gets dataset item from dataset.

    Args:
        dataset (uint32_hash_array): full size dataset in memory
        index (int): index

    Returns:
        List[int]: uint_32[16]
    """
    return dataset[x]

def calc_dataset_item(cache: uint32_hash_array, i: int) -> uint32_hash:
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

def get_full_size(block_number: int) -> int:
    sz = DATASET_BYTES_INIT + DATASET_BYTES_GROWTH * (block_number // EPOCH_LENGTH)
    sz -= MIX_BYTES
    while not isprime(sz / MIX_BYTES):
        sz -= 2 * MIX_BYTES
    return sz

def fnv(v1: uint64, v2: uint64) -> int:
    return ((v1 * FNV_PRIME) ^ v2) % 2**32

def get_seedhash(block_number: int) -> bytes:
    s = b'\x00' * 32
    for _ in range(block_number // EPOCH_LENGTH):
        s = serialize_hash(keccak_256(s))
    return s

def create_file_path(is_cache: bool, dir: str, block_nr: int) -> str:        
    dir = f'{dir}/' if dir[-1] != '/' else dir
    if not os.path.isdir(dir):
        os.mkdir(dir)
    first_8_bytes_seedhash = get_seedhash(block_nr).hex()[:16]
    file_path = f'{dir}{"cache" if is_cache else "full"}-R{REVISION}-{first_8_bytes_seedhash}'
    return file_path

# Assumes little endian bit ordering (same as Intel architectures)
def decode_int(s: bytes) -> int:
    """Decodes little endian encoded hex byte to little endian int

    Args:
        s (bytes): big endian encoded int

    Returns:
        int: little endian int
    """
    return int(s[::-1].hex(), 16) if s else 0

def encode_int(s: int) -> bytes:
    """Encodes int to little endian bytes

    Args:
        s (int): int to encode

    Returns:
        bytes: encoded int in little endian by
    """
    return s.to_bytes(4, 'little')

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
    """Takes uint32_hash_array and writes every entry as little endian bytes to file.

    Args:
        file_path (str): relative file path
        data (uint32_hash_array): uint_32[*][16]
    """
    with open(file_path, 'wb') as f:
        # write magic bytes
        f.write(b'\xad\xde\xe1\xfe\xfe\xca\xdd\xba')
        # write the real numbers
        for num_array in data:
            f.write(b''.join([int(num).to_bytes(4, 'little') for num in num_array]))
        
def read_from_file(file_path: str, element_amount: int) -> uint32_hash_array:
    """Reads 64byte hashes (little endian) into uint_32[*][16] array

    Args:
        file_path (str): relative path to file

    Returns:
        list: uint_32[*][16] array
    """
    with open(file_path, 'rb') as f:
        # jump over magic bytes
        f.read(8)
        # start filling array
        uint32_array = [[int.from_bytes(f.read(4), 'little') for __ in range(0, 64, 4)] for _ in range(element_amount)]
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