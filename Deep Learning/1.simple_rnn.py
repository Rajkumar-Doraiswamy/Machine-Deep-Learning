# Last amended: 27th June, 2019
# Myfolder: /home/ashok/Documents/8.rnn
# Ref: Page 196, Chapter 6, Deep Learning with Python, Francois Chollete
# For RNN diagrams, see:
#	https://towardsdatascience.com/recurrent-neural-networks-d4642c9bc7ce

#
# Objective:  Design a simple RNN from scratch

"""
Design of Simple rnn
No backpropogation
*********************


    ===========<==<= output@t ==<======<=======<========
    | state@t                                          ^
    \/                                                 |
==============>    |                                   |
dot(U,state@(t-1))  + ===>tanh() activation function >
==============>    |
dot(W,input@t)


How our two neural networks look like:

                        +                       +
                        +         U wts         +
                        +        =====>         +
    x1->        +       +          +            +
    x2->        +   W   +          +    W       +
    x3->        +       +          +            +
    x4->        +       +          +            +
                        +                       +
                        +                       +
                        +                       +

    |-----Ist NN,time= t ----|   |----IInd NN, time = t+1 -----|

Four input-features getting transformed to ten output-features
So number of wts =   W = output_features * input_features

Ten output features are concatenated to next ten
So number of wts =   U = output_features * output_features

Number of bias points are same as number of neurons in the
output layer = b

See details at the end how numpy dot product works
"""

%reset -f
import numpy as np

timesteps =100
input_features = 32
output_features = 64            # hidden layer

#1. Generate a timeseries having 100 timesteps and 32 features
#   This means that each row of 32 attributes is produced at
#   one time and next row at next timestep
inputs = np.random.random((timesteps, input_features))
# 1.1
inputs.shape                 # 100 , 32
# 1.2
inputs[0, : ]                # Features at 0th time
# 1.3
inputs[0,: ].shape           # 1D vector, (32,)


#2. Some initial state vector (one axis) 64 dimension
#    All with zero entries
state_t = np.zeros((output_features,))
state_t.shape                # 64
state_t


#3. Get arbitrary random weights, to start with
#   Weights from 'inputs' to 'outputs'
#   So for each feature, weights are column-wise
W = np.random.random((output_features, input_features))
# 3.1
W.shape          # (64,32) Compare result with that in #1.3
# 3.2 Weights from each 'state_t' to 'outputs'
U = np.random.random((output_features, output_features))
# 3.3
U.shape          # (64,64)
# 3.4   Get bias for each output neuron
b = np.random.random((output_features,))
W, U , b

# 3.5 Just how dot product would operate, see code at the end
#  NOTABLE THING IS EVEN THOUGH INPUT AT EACH TIMESTEP IS DIFFERENT
#      AND state_t LOOPS AROUND, WIGHTS W AND U ARE CONSTANT FOR 'EACH' RNN
#       IRRESPECTIVE OF TIMESTEPS.
#        THAT IS, ASSUMING YOU HAVE UNROLLED RNN INTO 'timesteps' RNN
#         WEIGHTS FOR EACH UNROLLED RNN REMAIN THE SAME.
output_t = np.tanh( np.dot(W, inputs[0]) + np.dot(U, state_t) + b )

# 3.6
output_t.shape

# 3.7 Just to see what input_t is like
#      before, we execute next step
for input_t in inputs:
    print(input_t.shape)        # It is (32,) ie 1D and not 2D
                                #  This observation is important


# 4. Initialise out output list
successive_outputs = []
for input_t in inputs:   # Each input is from one timestep; there are 100 of these
    # 4.1 For everyone of the timestep, get output
    #     How do we multiply a 2D matrix, ie W, with 1D array ie input_t?
    #     See below
    output_t = np.tanh( np.dot(W, input_t) + np.dot(U, state_t) + b )
    # 4.2 Store it as a list
    successive_outputs.append(output_t)
    # 4.3 This output becomes the next state_t
    state_t = output_t

# 4.4
len(successive_outputs)           # There are 100 lists of arrays,
                                  #   as many as timesteps

# 4.5
len(successive_outputs[0])        # Length of each array in the list is 64,
                                  #  same as number of output neurons

# 4.6
successive_outputs[:2]

# 4.7 Concatenate these arrays
final_output_sequence = np.concatenate(
                                      successive_outputs,
                                      axis = 0     # Each array in the list is 1D
                                                   # For arrays of dimension 1D, axis 1 is out of bounds
                                                   # Only option is axis = 0. This, sort of horizonatally
                                                   #  concatenates.
                                      )


# 4.7.1 Result of concatenation
final_output_sequence.shape            # (6400,)


# 4.8 So our RNN hidden-layer output of 100 timesteps is:
final_output_sequence.reshape(100,64)

##########################################################

"""
How to multiply 2D matrix with 1D array?
========================================
Ref: https://stackoverflow.com/a/54731529

    Such multiplication is outside the realm of matrix
    algebra. So how is it done?

    Multiplication of (3 X 2) with (2,1) is possible.
    But when we attempt multiplication of (3 X 2) with (2,)
    we are in a territory that's outside the jurisdiction of
    the usual rules of matrix multiplication. So, it's really
    up to the designers of the numpy.dot() function as to
    how the result turns out to be. They could've chosen
    to treat this as an error ("dimension mis-match"). Instead,
    they implemented the following solution:

"""

# a1.0 Our data. Two features and four observations
#      Each observation, ie row wise input is fed to NN.
#      Four observations are equivalent to 4-timesteps

data = np.array([[6,7 ],     #=> t = 0  [x1,x2]
                 [9,10],     #=> t = 1  [x1,x2]
                 [11,12],    #=> t = 2  [x1,x2]
                 [13,14]]    #=> t = 3  [x1,x2]
             )

data.shape     #  (4, 2)   OR 4-timesteps

# a2.0
# Output neurons = 3
# Input neurons  = 2
#
wt = np.array([[2,3],      # (w11,w21)        Two-Wts to output neuron1
               [3,4],      # (w12,w22)        Two-Wts to output neuron2
               [4,5]]      # (w13,w23)        Two-Wts to output neuron3
             )

wt.shape       # (3, 2)

# a3.0 Multiply as follows:
#      data[0] is first reshaped to data[0].reshape(2,1)
#      And then matrix-multiplication is performed

            6
            7
          ----
2   3       2 * 6 + 3 * 7
3   4       3 * 6 + 4 * 7
4   5       4 * 6 + 5 * 7

np.dot(wt,data[0])

# a4.0 Output of (3,1) is reshaped to (3,)

###############################################################
