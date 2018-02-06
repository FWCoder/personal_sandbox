import os.path
import random

filename = 'best4linreg.csv'
totalPointNumber = 1000
#startingNumber = random.randint(1, 10) # first random variable
#increaseValue = random.random() # second random variable
linear_coef = random.randint(1, 10)
count = 1

if os.path.exists(filename):
	os.remove(filename);

file = open(filename, 'w+')

while count < totalPointNumber :
	x1 = count
	x2 = random.random()
	y = x1*linear_coef + x2
	file.write(str(x1)+","+str(x2)+","+str(variable)+"\n")
	count += 1

file.close() 