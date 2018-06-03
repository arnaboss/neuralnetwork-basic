from numpy import exp, array, random, dot


class NeuralNetwork():
    def __init__(self):

        # Seed random number genrators, so it generates the same number everytime program runs
        random.seed(1)

        # We model a single neuron, with 3 input connections and 1 output connection
        # We assign random weights to a 3 x 1 matrix, in the range -1 to 1 and mean 0
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

    # The signmoid function which describes a S shaped curve
    # we pass a weighted sum of the inputs through this function
    # to normalize between 0 and 1
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # gradient of the sigmoid curve
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs,
              number_of_iterations):
        for _ in range(number_of_iterations):

            # Pass the training set through the neural network
            output = self.predict(training_set_inputs)

            # Calculate the error
            error = training_set_outputs - output

            # Multiply the error by the input and again by the gradient of the sigmoid curve
            adjustment = dot(training_set_inputs.T,
                             error * self.__sigmoid_derivative(output))

            # Adjust the weights
            self.synaptic_weights += adjustment

    def predict(self, inputs):
        # Pass inputs through the neural network (our single neuron)
        return self.__sigmoid(dot(inputs, self.synaptic_weights))


if __name__ == '__main__':

    # Initialize a neural network
    neural_network = NeuralNetwork()

    print('Random starting synaptic weights:', neural_network.synaptic_weights)

    # The training set. Four examples consisting of 3 input values and 1 output value
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    # Train the neural network using trainign set
    # Iterate 10k times and make small adjustments each time
    neural_network.train(training_set_inputs, training_set_outputs, 1000)

    print('New synaptic weights after training:',
          neural_network.synaptic_weights)

    # Test the neural network
    print('Considering new situation [1,0,0]) -> ?: ',
          neural_network.predict(array([1, 0, 0])))
