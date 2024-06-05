from pyomo.environ import *

# Create a model
model = ConcreteModel()

# Define decision variables
model.x = Var(initialize=1.0)
model.y = Var(initialize=1.0)

# Define objective function
model.obj = Objective(expr=model.x**2 + 2 * model.y**2, sense=minimize)

# Define constraints
model.constr1 = Constraint(expr=model.x + model.y == 1)

# Define solver
solver = SolverFactory("ipopt", executable="ipopt")

# Set solver options
solver.options["max_iter"] = 1000
solver.options["tol"] = 1e-6

# Solve the problem
solver.solve(model, tee=True)

# Display results
model.pprint()
