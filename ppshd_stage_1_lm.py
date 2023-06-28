# ppshd_stage_1_lm.py 
"""
Personnel and Patient Scheduling in the High Demanded Hospital Services: A Case Study in the Physiotherapy Service
By S. Noyan Ogulata & Melik Koyuncu & Esra Karakas

STAGE 1
Stage I: Patient Acceptance Planning
The purpose of this stage is to select patients that will be
scheduled for the following week from the candidate
patient list considering physiotherapist capacity and priority
of patients
"""

import ppshd_cfg

import pyomo.environ as pyo

model = pyo.AbstractModel(name="Personnel and Patient Scheduling")


# Parameter
# N : Number of patients
model.N = pyo.Param(within=pyo.PositiveIntegers)

def N_rule(model):
    return model.N >= 1

model.check_N = pyo.BuildCheck(rule=N_rule)

# N_set : set of patiens
model.N_set = pyo.RangeSet(model.N)

# Parameter
# S: Number of physiotherapists
model.S = pyo.Param(within=pyo.PositiveIntegers)

def S_rule(model):
    return model.S >= 1

model.check_S = pyo.BuildCheck(rule=S_rule)

# Parameter
# t[i] : Treatment time of i-th patient in minutes
model.t = pyo.Param(model.N_set, within=pyo.NonNegativeIntegers, initialize=0)


# Parameter
# T : Total daily capacity
model.T = pyo.Param(within=pyo.PositiveIntegers)

def T_rule(model):
    return model.T >= 1

model.check_T = pyo.BuildCheck(rule=T_rule)


# Parameter
# p[i] : Weight of priority level for i-th patient
model.p = pyo.Param(model.N_set, within=pyo.PositiveIntegers)
def p_rule(model, i):
    return model.p[i] >= 1

model.check_p = pyo.BuildCheck(model.N_set, rule=p_rule)


# Decision variable
# x[i] : 
#   1 if i-th patient is selected
#   0 otherwise
model.x = pyo.Var(model.N_set, within=pyo.Binary, initialize=0)


# OBJETIVE FUNCTION AND CONSTRICTIONS ---------------
# Maximize number of patients priority and time
def obj_rule(model):
    return pyo.sum_product(model.p, model.x)

model.OBJ= pyo.Objective(rule=obj_rule, sense=pyo.maximize)

# Constraint 1: SUM x * t <= T
def enough_time_rule(model):
    return pyo.sum_product(model.x, model.t, index=model.N_set) <= model.T

model.enough_time = pyo.Constraint(rule=enough_time_rule)


opt = pyo.SolverFactory('glpk')
# opt = pyo.SolverFactory('cplex')

instance = model.create_instance(ppshd_cfg.LM_STAGE_1_DAT_FILE)


results = opt.solve(instance, load_solutions=False)  # solves and updates instance
# @:tail


def run_solver():
    """
    Run solver and get solution
    """
    
    # import pyomo.environ as pyo

    print(instance.name, '\n')

    from pyomo.opt import SolverStatus, TerminationCondition
    # To avoid automatic loading of the solution from the results object to the model, use the load solutions=False argument to the call to solve().
    if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
        # Manually load the solution into the model
        print("Tenemos solucion optima")
        instance.solutions.load_from(results)
    else:
        print("Solve failed.")
        # instance.solutions.load_from(results)


run_solver()

def print_cosas():
    total_time = 0
    total_patients = 0
    patients_indices = []
    patients_paper_indices = [1, 2, 3, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 28, 29, 30, 34, 35, 38, 41, 42, 47, 48, 49, 50, 51, 53, 54, 56, 60, 61, 62, 63, 64, 65, 66, 69, 71, 72, 74, 75, 76, 79, 80, 85, 86, 87]
    total_time_paper = 0
    print(instance.name)
    # instance.T.pprint()
    # instance.N.pprint()
    # instance.N_set.pprint()
    for i in instance.N_set:
        if pyo.value(instance.x[i]) != 0:
            total_time += pyo.value(instance.x[i]) * pyo.value(instance.t[i])
            # print("x[", i, "]=", pyo.value(instance.x[i]), ",, t[", i, "]=", pyo.value(instance.t[i]), ",, total_time =", total_time)
            total_patients += 1
            patients_indices.append(i)

    for i in patients_paper_indices:
        total_time_paper += pyo.value(instance.x[i]) * pyo.value(instance.t[i])

    print("RESULTS WITH SOLVER --------------------")
    print("patients_indices =", patients_indices)
    print("total_patients =", total_patients)
    print("total_time =", total_time)
    print("model.OBJ =", pyo.value(instance.OBJ))

    print("RESULTS IN PAPER -----------------------")
    print("total_patients paper =", len(patients_paper_indices))
    print("total_time_paper =", total_time_paper)
    paper_obj = 0
    for i in instance.N_set:
        paper_obj += pyo.value(instance.p[i]) * pyo.value(instance.x[i])
    print("PAPER.OBJ =", paper_obj)
print_cosas()
