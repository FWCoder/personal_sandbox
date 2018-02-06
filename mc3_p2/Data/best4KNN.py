import os.path
import random
import numpy as np
import math

def poly_function(x, polynomial_rate):
	variable = 0
	current_polynomial_rate = polynomial_rate
	while(current_polynomial_rate >= 0):
		variable += random.randint(-10,10)*pow(random.randint(-10,10), current_polynomial_rate)
		current_polynomial_rate -= 1
	return variable

filename = 'best4KNN.csv'
totalPointNumber = 1000
#polynomial_rate = random.randint(1,5)
#startingNumber = 1.545
#increaseValue = random.randrange(1, 1000, 1.4)
count = 1

if os.path.exists(filename):
	os.remove(filename);

file = open(filename, 'w+')
a = random.randint(1,10)	

while count < totalPointNumber :
	#startingNumber += increaseValue
	# variable = 0
	# current_polynomial_rate = polynomial_rate
	# while(current_polynomial_rate >= 0):
	# 	variable += random.randint(-10,10)*pow(random.randint(-10,10), current_polynomial_rate)
	# 	current_polynomial_rate -= 1

  	#first_dimension = random.randint(0,100)
  	#first_dimension = random.uniform(-1, 1)
  	#second_dimension = random.randint(1,100)
  	#random.seed(1234)
  	x1 = random.randint(0,100)
  	x2 = random.randint(1,100)
  	
	variable = np.cos(x1/a + x2/(a**2))
	file.write(str(x1)+","+str(x2)+","+str(variable)+"\n")
	#variable = np.sin(count*np.pi/180.)
	#variable = random.uniform(-1, 1)

	# if(count != totalPointNumber - 1):
	# 	tempVal = count%3.0
	# 	if(tempVal == 0.0 and count != 0.0):
	# 		file.write(str(variable)+"\n")
	# 	else:
	# 		file.write(str(variable)+",");
	# else:
	# 	file.write(str(variable));

	
	count += 1

file.close() 