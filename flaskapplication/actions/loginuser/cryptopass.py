import base64
from Crypto.Cipher import AES
from Crypto.Hash import SHA256
from Crypto import Random
import random
import string

class passcrypt:

    def passwordencrypt(self, orgpass):
        salt = ''.join(random.choices(string.ascii_uppercase + string.digits, k=35))
        salt = salt.encode('utf-8')
        orgpass = bytes(orgpass, encoding='utf-8')
        key = SHA256.new(salt).digest()  # use SHA-256 over our key to get a proper-sized AES key
        IV = Random.new().read(AES.block_size)  # generate IV
        encryptor = AES.new(key, AES.MODE_CBC, IV)
        padding = AES.block_size - len(orgpass) % AES.block_size  # calculate needed padding
        orgpass += bytes([padding]) * padding  # Python 2.x: source += chr(padding) * padding
        data = IV + encryptor.encrypt(orgpass)  # store the IV at the beginning and encrypt
        return base64.b64encode(data).decode("latin-1"), salt.decode('utf-8')

    def passworddecrypt(self, salt, enpass):
        salt = salt.encode('utf-8')
        enpass = base64.b64decode(enpass.encode("latin-1"))
        key = SHA256.new(salt).digest()  # use SHA-256 over our key to get a proper-sized AES key
        IV = enpass[:AES.block_size]  # extract the IV from the beginning
        decryptor = AES.new(key, AES.MODE_CBC, IV)
        data = decryptor.decrypt(enpass[AES.block_size:])  # decrypt
        padding = data[-1]  # pick the padding value from the end; Python 2.x: ord(data[-1])
        temp = data[:-padding]
        if data[-padding:] != bytes([padding]) * padding:  # Python 2.x: chr(padding) * padding
            raise ValueError("Invalid padding...")
        decodedpass = temp.decode('utf-8')
        return decodedpass  # remove the padding