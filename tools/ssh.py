import os
import paramiko
import configparser

class MySFTPClient(paramiko.SFTPClient):
    def put_dir(self, source, target):
        for item in os.listdir(source):
            if os.path.isfile(os.path.join(source, item)):
                print(20*'*', '%s/%s' % (target, item), 20*'*')
                self.put(os.path.join(source, item), '%s/%s' % (target, item))
            else:
                self.mkdir('%s/%s' % (target, item), ignore_existing=True)
                self.put_dir(os.path.join(source, item), '%s/%s' % (target, item))
    def mkdir(self, path, mode=511, ignore_existing=False):
        try:
            super(MySFTPClient, self).mkdir(path, mode)
        except IOError:
            if ignore_existing:
                pass
            else:
                raise Exception('director is exist')

if __name__ == '__main__':
    Config = configparser.ConfigParser()
    Config.read('./pandasquant/config.ini')
    HOST = Config['host']['HOST']
    PORT = Config['host']['PORT']
    USERNAME = Config['host']['USERNAME']
    KEY = paramiko.RSAKey.from_private_key_file(Config['host']['PKEY_FILE'])
    source_path = Config['host']['source_path']
    target_path = Config['host']['target_path']
    
    transport = paramiko.Transport((HOST, int(PORT)))
    transport.connect(username=USERNAME, pkey=KEY)
    stfp = MySFTPClient.from_transport(transport)
    stfp.mkdir(target_path, ignore_existing=True)
    stfp.put_dir(source_path, target_path)
    stfp.close()