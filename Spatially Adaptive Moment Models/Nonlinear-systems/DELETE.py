import configparser

config = configparser.ConfigParser()

config.read('config.txt')
print(config.sections())

numerical_method_information = config['numerical_method_information']

ordersList = numerical_method_information['orders']
print(ordersList)
ordersList = [int(order) for order in ordersList.split(',')]

print(ordersList)

