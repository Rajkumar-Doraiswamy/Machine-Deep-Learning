# Last amended: 25th May, 2019
# Myfolder: /home/ashok/Documents/2. data_augmentation
# VM: lubuntu_deeplearning
# Ref: Page 136, Chapter 5, Deeplearning with Python by Fracois Chollet
#      https://stackoverflow.com/questions/29864366/difference-between-function-and-generator

# Objective: What is a python generator?
#            Generator helps to create an
#            iterator, which when called
#            returns, an object that you
#            might have designed

# Example 1
# 1.
def mygenerator():
    i = 0
    # 1.2
    while True:
        i += 1
        yield i     # 'yield' returns a value
                    # Unlike in return statement
                    # function is not terminated.
                    # This statement pauses the
                    # function saving all its states
                    # and later continues from there
                    # on successive calls.

# 2
for item in mygenerator():
    # 2.1
    print(item)
    # 2.2
    if item >=4:
        break

# 3.1  Another way of using generator using next()
ab = mygenerator()     # Generator returns an iterator.
ab                     # So 'ab' is an iterator
# 3.2 Start iterating
next(ab)
next(ab)

# 3.3 Or use iterator, as
for i in ab:
    print(i)
    if i > 20:
        break



# 3.4 A generator that takes an argument and
#     starts from there
def arggen(st):
    while True:
        st = st * 2
        yield st

# 3.5
t = arggen(4)
next(t)
next(t)



# 4.
"""
What is a Generator?
    4.1 Simply speaking, a generator is a function that
        returns an object (iterator) which we can iterate
        over (one value at a time).

    4.2 A generator returns an iterator. It is an Object
#       one can use in 'for..in' loop. It uses 'yield'
#       to return the value.
#   4.3 So what is the difference between 'yield' and 'return'
#       each time generator() is called in the for-loop,
#       it remembers its earlier state: this is because of
#       'yield'. A 'return' does not remember the earlier
#        state.
#  4.4 In short, a generator looks like a function but
#      behaves like an iterator.

"""

"""
So what would an image datagenerator
would look like?

    def imagedatagenr(image):
        while True:
            Process image randomly
            yield image

But in sklearn we have to maintain
some discipline. And also we have
some constraints. So we use fit()
and flow() functions, as:

Step1: Returns a learning object
dg = ImageDataGenerator(
                        What would be done to an image
                       )

Step2: Learn
dg.fit(X_train,y_train)


Step3: Transform and return
dg.flow(X_train,
        y_train,
        batch_size = 32
        )

Step4: Apply same learning on X_test
dg.flow(X_test,
        y_test,
        batch_size = 32
        )


"""





#################
# 5. Another example
#    https://realpython.com/introduction-to-python-generators/
#################


# 5.1 Execution begins at the start of the function
#     When calling next() the first time,
#     body and continues until the next yield statement
#     where the value to the right of the statement is returned,
#     subsequent calls to next() continue from the yield statement
#     to the end of the function, and loop around and continue from
#     the start of the function body until another yield is called.

# 5.2
def countdown(num):
    print('Starting')
    i = 0
    while num > 0:
        i = i+1          # Note that even value of 'i' will be remembered
        print(i)         #  between calls even though it is not 'yielded'
        yield num
        num -= 1

# 5.3
val = countdown(5)
val

# 5.4
next(val)
# 5.5
next(val)



########## I am done here ###################


#################
# 6. Third example
#################

from itertools import count

# itertools provide count to generate infinite stream
#  of integer. You can give start and step to tell starting
#   and stepping value for the generated stream. I am going
#    to use this in the following example.

# 6.1
for i in count(start=0, step=1):
    print(i)

# 6.2 A simple example to generate list of even numbers.
#     Build and return a list:

def find_even_number_generator(number_stream):
    for n in number_stream:
        if n % 2 == 0:
            yield n

# 6.3
for i in find_even_number_generator(count()):
    print(i)

#################################
