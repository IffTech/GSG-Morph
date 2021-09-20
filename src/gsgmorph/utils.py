class IncompatibleGraphError(Exception):
    """Exception used when graphs are given to generate a QUBO that
    fail to satisfy or violate the conditions needed to generate the
    QUBO.

    Args:
        message (str):
            String telling the user why the graphs given cannot be used
            to generate a certain QUBO.
    """

    def __init__(self, message):
        self.message = message
