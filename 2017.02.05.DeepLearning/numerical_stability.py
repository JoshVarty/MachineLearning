oneBillion = 1000000000
currentNum = oneBillion

for i in xrange(1, 1000000):
    currentNum = currentNum + 0.000001

currentNum = currentNum - oneBillion
print currentNum

#0.953673362732