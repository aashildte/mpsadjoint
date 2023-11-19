import dolfin as df
import dolfin_adjoint as da


class NonlinearProblem(df.NonlinearProblem):
    """

    Class extending dolfin's NonLinearProblem class,
    makes it possible to save matrices instead if reinitializing
    them for every iteration.

    Args:
        J: bilinear form - RHS
        F: linear form - LHS
        bcs - boundary conditions (list; can be empty)

    """

    def __init__(self, R, state, bcs):
        J = df.fem.formmanipulations.derivative(R, state)
        self.bilinear_form = J
        self.linear_form = R
        self.bcs = bcs
        self.n = 0
        df.NonlinearProblem.__init__(self)

    def F(self, b, x):
        """

        Assembles linear form (LHS)

        Args:
            b - vector of linear system
            x - ?

        """
        da.assemble(self.linear_form, tensor=b)
        
        if isinstance(self.bcs, list):
            for bc in self.bcs:
                bc.apply(b, x)
        else:
            self.bcs.apply(b, x)
    
    def J(self, A, x):
        """

        Assembles bilinear form (RHS)

        Args:
            A - matrix of linear system
            x - ?

        """
        da.assemble(self.bilinear_form, tensor=A)
        
        if isinstance(self.bcs, list):
            for bc in self.bcs:
                bc.apply(A)
        else:
            self.bcs.apply(A)
