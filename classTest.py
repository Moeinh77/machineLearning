class first(object):

	def function1(self):
	    print('first')
	
	def __str__(self):
		return "first"


class second(first):
	
	 def __init__(self):
	        super(second,self).__init__()

	 def function2(self):
		self.function1()
		print('second')
	
	 def __str__(self):
      	   return super(second,self).__str__() 


x=second()
print(x)
	
