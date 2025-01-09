import configparser

config = configparser.ConfigParser()

config.read('config.txt')
print(config.sections())