import math

from neurgen.utilities import rand_weight, check_bounds_limits

class ProtoNode(object):
    """
    This class is the prototype for nodes.  Nodes are the holder of values,
    they activate and they maintain connnections to other nodes.

    """

    def __init__(self):
        self.node_no = None
        self.node_type = None
        self._value = 0.0
        self.input_connections = []
        self._activation_type = None
        self.error = 0.0
        self.target = None

    def get_value(self):
        """
        This function returns the value of the node.  This is the value prior
        to activation.

        """

        return self._value

    @staticmethod

    def _activate(value):
        """
        This is a stub function.  Activations will vary by node.

        """

        return value

    @staticmethod

    def _error_func(value):
        """
        This is a stub function.

        """

        return value

    def activate(self):
        """
        This function applies the activation function to the value of the node.

        """

        return self._activate(self._value)

    def error_func(self, value):
        """
        This function computes the error function, typically the derivative of
        the error.

        """

        return self._error_func(value)

    def randomize(self, random_constraint=1.0):
        """
        This function assigns a random value to the input connections.
        The random constraint limits the scope of random variables.

        """

        for conn in self.input_connections:
            conn.set_weight(rand_weight(random_constraint))

    def get_activation_type(self):
        """
        This function returns the activation type of the node.

        """

        return self._activation_type

    def update_error(self, halt_on_extremes):
        """
        This function updates the error of the node from upstream errors.

        Depending upon halting on extremes, it also may adjust or halt if
        overflows occur.

        Finally, it computes the derivative of the activation type, and
        modifies the error.

        """
        if self.node_type == 'output':
            self.error = self.target - self.activate()

        #   Other than output layer, will have accumulated errors from
        #   above
        self.error *= check_bounds_limits(
            self.error_func(self.activate()),
            halt_on_extremes)

        self._update_lower_node_errors(halt_on_extremes)

    def _update_lower_node_errors(self, halt_on_extremes):
        """
        This function goes through each of the input connections to the node
        and updates the lower nodes.

        The error from the current node is multiplied times the connection
        weight, inspected for bounds limits and posted in the lower node's
        error.

        """

        for conn in self.input_connections:
            conn.lower_node.error += check_bounds_limits(
                conn.get_weight() * self.error,
                halt_on_extremes)


class Node(ProtoNode):
    """
    This class implements normal nodes used in the network.  The node type is
    specified, and must be in ['sigmoid', 'tanh', 'linear'].

    """

    def __init__(self, node_type=None):
        ProtoNode.__init__(self)
        self.node_type = node_type
        self._error_func = None

    def set_activation_type(self, activation_type):
        """
        This function sets the activation type for the node.  Currently
        available values are 'sigmoid', 'tanh', 'linear'.  When specifying the
        activation type, the corresponding derivative type for the error
        functions are assigned as well.

        """

        if activation_type == 'sigmoid':
            self._activate = sigmoid
        elif activation_type == 'tanh':
            self._activate = tanh
        elif activation_type == 'linear':
            self._activate = linear
        else:
            raise ValueError("invalid activation type: %s" % (activation_type))

        self._set_error_func(activation_type)
        self._activation_type = activation_type

    def _set_error_func(self, activation_type):
        """
        This function sets the error function type.

        """

        if activation_type == 'sigmoid':
            self._error_func = sigmoid_derivative
        elif activation_type == 'tanh':
            self._error_func = tanh_derivative
        elif activation_type == "linear":
            self._error_func = linear_derivative
        else:
            raise ValueError("Invalid activation function")

    def set_value(self, value):
        """
        Set value used to avoid the accidental use of setting a value on a
        bias node.  The bias node value is always 1.0.

        """

        self._value = value

    def get_value(self):
        """
        This function returns the internal value of the node.

        """

        return self._value

    def feed_forward(self):
        """
        This function walks the input connections, summing gets the lower node
        activation values times the connection weight.  Then, node is
        activated.

        """

        sum1 = 0.0
        for conn in self.input_connections:
            if conn.lower_node.get_value() is None:
                raise ValueError("Uninitialized node %s" % (
                            conn.lower_node.node_no))

            sum1 += conn.lower_node.activate() * conn.get_weight()

        self.set_value(sum1)

    def add_input_connection(self, conn):
        """
        This function adds an input connection.  This is defined as a
        connection that comes from a layer on the input side, or in this
        applicaion, a lower number layer.

        The reason that there is a specific function rather than using just an
        append is to avoid accidentally adding an input connection to a bias
        node.

        """
        if conn.upper_node == self:
            self.input_connections.append(conn)
        else:
            raise ValueError("The upper node is always current node.")

    def adjust_weights(self, learnrate, halt_on_extremes):
        """
        This function adjusts incoming weights as part of the back propagation
        process, taking into account the node error.  The learnrate moderates
        the degree of change applied to the weight from the errors.

        """

        for conn in self.input_connections:
            conn.add_weight(
                check_bounds_limits(
                    self._adjust_weight(
                        learnrate,
                        conn.lower_node.activate(),
                        self.error),
                    halt_on_extremes))
            #   Fix this
            conn.weight_adjusted = True

            #   Overflow check on weight
            conn.set_weight(
                        check_bounds_limits(
                            conn.get_weight(),
                            halt_on_extremes))

    @staticmethod

    def _adjust_weight(learnrate, activate_value, error):
        """
        This function accepts the learn rate, the activated value received
        from a node connected from below, and the current error of the node.

        It then multiplies those altogether, which is an adjustment to the
        weight of the connection as a result of the error.

        """

        return learnrate * activate_value * error


class CopyNode(Node):
    """
    This class maintains the form used for copy nodes in recurrent networks.
    The copy nodes are used after propagation.  The values from nodes in upper
    layers, such as the hidden nodes are copied to the CopyNode.  The
    source_node defines the node from where the value arrives.

    An issue with using copy nodes, is that you must be careful to
    adhere to a sequence when using the nodes.  For example, if a copy node
    value is a source to another copy node, you will want to copy the values
    from downstream nodes first.

    """

    def __init__(self):
        Node.__init__(self)
        self.node_type = 'copy'
        self._source_node = None
        self._source_type = None
        self._incoming_weight = 1.0
        self._existing_weight = 0.0

        self.set_activation_type('linear')

    def set_source_node(self, node):
        """
        Sets the source of previous recurrent values.

        """

        self._source_node = node

    def get_source_node(self):
        """
        Gets the source of previous recurrent values.

        """

        return self._source_node

    def load_source_value(self):
        """
        This function transfers the source node value to the copy node value.

        """
        if self._source_type == 'a':
            value = self._source_node.activate()
        elif self._source_type == 'v':
            value = self._source_node.get_value()
        else:
            raise ValueError("Invalid source type")

        self._value = self._value * self._existing_weight + \
                        value * self._incoming_weight

    def get_source_type(self):
        """
        This function gets the type of source value to use.

        Source type will be either 'a' for the activation value or 'v' for the
        summed input value.

        """

        return self._source_type

    def get_incoming_weight(self):
        """
        This function gets the value that will be multiplied times the
        incoming source value.

        """

        return self._incoming_weight

    def get_existing_weight(self):
        """
        This function gets the value that will be multiplied times the
        existing value.

        """

        return self._existing_weight

    def source_update_config(self, source_type, incoming_weight,
                        existing_weight):
        """
        This function accepts parameters governing what the source information
        is used, and how the incoming and existing values are discounted.

        Source type can be either 'a' for the activation value or 'v' for the
        summed input value.

        By setting the existing weight to zero, and the incoming discount to
        1.0. An Elman style update takes place.

        By setting the existing weight to some fraction of 1.0 such as .5, a
        Jordan style update can take place.

        """

        if source_type in ['a', 'v']:
            self._source_type = source_type
        else:
            raise ValueError(
                "Invalid source type, %s. Valid choices are 'a' or 'v'")

        errmsg = """The incoming weight, %s must be a float value
                    from 0.0 to 1.0""" % (incoming_weight)
        if not isinstance(incoming_weight, float):
            raise ValueError(errmsg)
        if not (0.0 <= incoming_weight <= 1.0):
            raise ValueError(errmsg)

        self._incoming_weight = incoming_weight

        errmsg = """The existing_weight, %s must be a float value
                    from 0.0 to 1.0""" % (existing_weight)
        if not isinstance(existing_weight, float):
            raise ValueError(errmsg)
        if not (0.0 <= existing_weight <= 1.0):
            raise ValueError(errmsg)

        self._existing_weight = existing_weight


class BiasNode(ProtoNode):
    """
    Bias nodes provide value because of their connections, and their value and
    activation is always 1.0.

    """

    def __init__(self):
        ProtoNode.__init__(self)
        self.node_type = 'bias'
        # value is always 1.0, it's the connections that matter
        self._value = 1.0
        self._activated = self._value

    @staticmethod

    def activate(value=None):
        """
        The activation of the bias node is always 1.0.

        """

        return 1.0

    @staticmethod

    def error_func(value=None):
        """
        The activation of the bias node is always 1.0.  Value is ignored, but
        left in for consistency with other nodes.

        """

        return 1.0

class Connection(object):
    """
        Connection object that holds the weighting information between nodes
        as well as a reference to the nodes that are connected.

    """

    def __init__(self, lower_node, upper_node, weight=0.0):
        """
        The lower_node lives on a lower layer, closer to the input layer.
        The upper mode lives on a higher layer, closer to the output layer.

        """

        self.lower_node = lower_node
        self.upper_node = upper_node
        self._weight = None
        self.set_weight(weight)

    def set_weight(self, weight):
        """
        This function sets the weight of the connection, which relates to
        the impact that a lower node's activation will have on an upper node's
        value.

        """

        err_msg = "The weight, %s, must be a float value" % (weight)
        if not isinstance(weight, float):
            raise ValueError(err_msg)
        else:
            self._weight = weight

    def add_weight(self, weight):
        """
        This function adds to the weight of the connection, which is
        proportional to the impact that a lower node's activation will
        have on an upper node's value.

        """

        err_msg = "The weight, %s, must be a float value" % (weight)
        if not isinstance(weight, float):
            raise ValueError(err_msg)
        else:
            self._weight += weight

    def get_weight(self):
        """
        This function sets the weight of the connection, which is relates to
        the impact that a lower node's activation will have on an upper node's
        value.

        """

        return self._weight


def sigmoid(value):
    """
    Calculates the sigmoid .

    """

    try:
        value = 1.0 / (1.0 + math.exp(-value))
    except OverflowError:
        value = 0.0

    return value


def sigmoid_derivative(value):
    """
    Calculates the derivative of the sigmoid for the value.

    """

    return value * (1.0 - value)


def tanh(value):
    """
    This function calculates the hyperbolic tangent function.

    """

    return math.tanh(value)


def tanh_derivative(value):
    """
    This function calculates the tanh derivative of the value.

    """

    return 1.0 - pow(math.tanh(value), 2)


def linear(value):
    """
    This function simply returns the value given to it.

    """

    return value


def linear_derivative(value):
    """
    This function returns 1.0.  Normally, I would just return 1.0, but pylint
    was complaining.

    """

    value = 1.0
    return value

