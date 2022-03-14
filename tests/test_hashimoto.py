import unittest
import ethash

class TestHashimoto(unittest.TestCase):
    def test_chashimoto_light(self):
        # TODO remove cache before
        print('Creating cache...')
        byteArray = ethash.create_cache(30000, './ethash_data')
        print('Creating hash...')
        result = ethash.hashimoto_light(bytearray.fromhex('76a35fb221d2f471375c7db942b0513b9d0e2589711988cfb2ecdfaf5e171ba3'),
                                    bytearray.fromhex('a8e9c2db86671707'),
                                    30000, byteArray)
        print('Checking the hashes...')
        self.assertEqual(result['mix digest'].hex(), 'dc0b2bc9e4f3fd8ce5d0d971a78d035fa964cf5714ae772990fe4d4562902dba')
        self.assertEqual(result['result'].hex(), '00000000006e9cf98ef9b16205caa91f0891c292b252b9081e8f7ef55ecd445f')
        print('Done.')
    # TODO: create test case reading cache from disk
        
if __name__ == '__main__':
    unittest.main()