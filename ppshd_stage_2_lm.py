# ppshd_stage_2_lm.py 
"""
Personnel and Patient Scheduling in the High Demanded Hospital Services: A Case Study in the Physiotherapy Service
By S. Noyan Ogulata & Melik Koyuncu & Esra Karakas

STAGE 2
Stage II: Patient Acceptance Planning.
Assignment to Physiotherapists.
The selected patients at the first stage must be assigned
to physiotherapists at this stage.
"""

import ppshd_cfg

import pyomo.environ as pyo

model = pyo.AbstractModel(name="Personnel and Patient Scheduling. Stage II")


# Parameter
# n: Number of selected patients
model.n = pyo.Param(within=pyo.PositiveIntegers)

def n_rule(model):
    return model.n >= 1

model.check_n = pyo.BuildCheck(rule=n_rule)

# Set
# n_set : set of selected patients
model.n_set = pyo.Set(within=pyo.NonNegativeIntegers)

# Parameter
# t[i] : Treatment time of i-th patient in minutes of selected patients
model.t = pyo.Param(model.n_set, within=pyo.NonNegativeIntegers, initialize=0)

# Parameter
# S: Number of physiotherapists
model.S = pyo.Param(within=pyo.PositiveIntegers)

def S_rule(model):
    return model.S >= 1

model.check_S = pyo.BuildCheck(rule=S_rule)

# Set
# S_set : set of physiotherapists
model.S_set = pyo.RangeSet(model.S)


# Parameter
# h: Daily work minutes per each physiotherapist
model.h = pyo.Param(within=pyo.NonNegativeIntegers)

def h_rule(model):
    return model.h >= 0

model.check_h = pyo.BuildCheck(rule=h_rule)


# Set
# W_set : Set of penalty weights coefficients
model.W_set = pyo.Set(within=pyo.PositiveIntegers)

# Parameter
# W[w] : Penalty weights coefficients in the objective function
# assigned to each goal prescribed.
# Determination of these weights depends on the users preferences
model.W = pyo.Param(model.W_set, within=pyo.PositiveIntegers)

# Set
# K_set : Time category number.
# 1, and 2 were assigned to symbolize 
# short-term (0–39 min), long-term (>40 min) respectively
model.K_set = pyo.Set(within=pyo.PositiveIntegers)

# Parameter
# tc[i] : Time category for ith patients. (1 = short, 2 = long)
# Create tc
def tc_init(model, i):
    if model.t[i] < 40:
        return 1
    else:
        return 2
    
model.tc = pyo.Param(model.n_set, initialize=tc_init)



# VARIABLES -----------------------------------------------

# variables
# G[j] : Total physiotherapy time assigned to j-th physiotherapist
model.G = pyo.Var(model.S_set, within=pyo.NonNegativeIntegers, initialize=0)
#
# G_ave : Average physiotherapy time assigned to physiotherapists
model.G_ave = pyo.Var(within=pyo.NonNegativeReals, initialize=0)
#
# d_range : number of d_plus variables to objetive function
# model.d_plus_range = pyo.RangeSet(4)
model.d_plus_range = pyo.Set(initialize=model.W_set[:-1])
#
# d_plus : Goals in objetive function
#   d_plus[1] : represents the absolute deviation 
#       from the goal of balanced time distribution among physiotherapists
#
#   d_plus[2] : absolute deviations 
#       from the goal of balanced distribution of patients 
#       in terms of time categories for short operations 
#
#    d_plus[3] : absolute deviations 
#       from the goal of balanced distribution of patients 
#       in terms of time categories for long operations
# 
#   d_plus[4] : TODO GIVE AN EXPLANATION TO THIS VARIABLE. I DONT KNOW WHAT IT IS
model.d_plus = pyo.Var(model.d_plus_range, within=pyo.NonNegativeReals, initialize=0)
#
# d_minus_4 : is the deviation terms related to
# loading physiotherapists below their daily capacitie
model.d_minus_4 = pyo.Var(within=pyo.Reals, initialize=0)
#
# NP[jk] : Number of patients assigned to j-th physiotherapist from k-th time category
model.NP = pyo.Var(model.S_set, model.K_set, within=pyo.Binary, initialize=0)
#
# NP_ave[k] : Average number of patients assigned from k-th time category.
model.NP_ave = pyo.Var(model.K_set, within=pyo.NonNegativeReals, initialize=0)
#
#
# Decision variable
# y[ij] : 
#   1 if i-th patient is assigned to j-th physiotherapist
#   0 otherwise
model.y = pyo.Var(model.n_set, model.S_set, within=pyo.Binary, initialize=0)

# --------------------------------------------------------
# --------------------------------------------------------

# OBJETIVE FUNCTION AND CONSTRICTIONS ---------------
# Maximize number of patients priority and time
def obj_rule(model):
    # return model.W[1] * model.d_plus[1] + model.W[2] * model.d_plus[2] + model.W[3] * model.d_plus[3] + model.W[4] * model.d_plus[4] + model.W[5] * model.d_minus_4
    return pyo.sum_product(model.W, model.d_plus, index=model.W_set) + model.W[5] * model.d_minus_4

model.OBJ= pyo.Objective(rule=obj_rule, sense=pyo.minimize)

# Constraint. Eq (4): d_plus[1] : absolute time deviation
def d_plus_1_rule(model):
    # TODO : FALTA VALOR ABSOLUTO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # FALTA VALOR ABSOLUTO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    return model.d_plus[1] == pyo.sum(model.G[j] - model.G_ave for j in model.S_set)

model.d_plus_1_costr = pyo.Constraint(rule=d_plus_1_rule)

# Constraint. Eq (5): G[j] : Total physiotherapy time assigned to jth physiotherapist
def Gj_costr_rule(model, j):
    return model.G[j] == pyo.sum(model.t[i] * model.y[i,j] for i in model.n_set)

model.Gj_costr = pyo.Constraint(model.S_set, rule=Gj_costr_rule)

# Constraint. Eq (6): G_ave : Average physiotherapy time assigned to physiotherapists
def G_ave_costr_rule(model):
    return model.G_ave == pyo.sum(model.G[j]/model.S for j in model.S_set)

model.G_ave_costr = pyo.Constraint(rule=G_ave_costr_rule)

# Constraint. Eq (7): d_plus[2], d_plus[3] : 
#   Absolute deviations from the goal of balanced
#   distribution of patients in terms of time categories,
#   respectively for short and long operations 
def d_plus_23_rule(model, k):
    # TODO : FALTA VALOR ABSOLUTO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # FALTA VALOR ABSOLUTO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    return model.d_plus[k+1] == pyo.sum(model.NP[j,k] * model.NP_ave[k] for j in model.S_set)

model.d_plus_23_costr = pyo.Constraint(model.K_set, rule=d_plus_23_rule)

# Constrainr. Eq (8): NP[j,k] : Number of patients assigned to j-th physiotherapist from k-th time category
def NP_rule(model):
    return model.N[j,k] == pyo.sum()

model.NP_costr = pyo.Constraint(model.K_set, rule=NP_rule)

# ESTAS AQUI    # ESTAS AQUI    # ESTAS AQUI    # ESTAS AQUI    
# ESTAS AQUI    # ESTAS AQUI    # ESTAS AQUI    # ESTAS AQUI    


# SOLVE ABSTRACT MODEL ------------------------------------

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
