# ppshd_stage_2_lm.py 
"""
Personnel and Patient Scheduling in the High Demanded Hospital Services: A Case Study in the Physiotherapy Service
By S. Noyan Ogulata, Melik Koyuncu, Esra Karakas

STAGE 2
Stage II: Patient Acceptance Planning.
Assignment to Physiotherapists.
The selected patients at the first stage must be assigned
to physiotherapists at this stage.
"""

import ppshd_cfg

import pyomo.environ as pyo

model = pyo.AbstractModel(name="Personnel and Patient Scheduling. Stage II")

# BEGIN : Data from ppshd_stage_1_lm ----------------------
# Parameter
# N: Number of initial patients
model.N = pyo.Param(within=pyo.PositiveIntegers)

# Parameter
# T : Total daily capacity in minutes
model.T = pyo.Param(within=pyo.PositiveIntegers)

# N_set : set of patients
model.N_set = pyo.RangeSet(model.N)

# Parameter
# p[i] : Weight of priority level for i-th patient
model.p = pyo.Param(model.N_set, within=pyo.PositiveIntegers)
def p_rule(model, i):
    return model.p[i] >= 1
model.check_p = pyo.BuildCheck(model.N_set, rule=p_rule)

# Parameter
# t[i] : Treatment time of i-th patient in minutes
model.t = pyo.Param(model.N_set, within=pyo.PositiveIntegers)
# END : Data from ppshd_stage_1_lm ------------------------


# Parameter
# n: Number of selected patients
model.n = pyo.Param(within=pyo.PositiveIntegers)
def n_rule(model):
    return model.n >= 1
model.check_n = pyo.BuildCheck(rule=n_rule)

# Set
# n_set : set of selected patients
model.n_set = pyo.Set(within=pyo.PositiveIntegers)

# # Parameter
# # t[i] : Treatment time of i-th patient in minutes of selected patients
# model.t = pyo.Param(model.n_set, within=pyo.NonNegativeIntegers, initialize=0)

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
    return model.h > 0
model.check_h = pyo.BuildCheck(rule=h_rule)


# Parameter
# c: Maximum number of patients assigned to each physiotherapist
model.c = pyo.Param(within=pyo.PositiveIntegers)
def c_rule(model):
    return model.c > 0
model.check_c = pyo.BuildCheck(rule=c_rule)


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
# 1 and 2 were assigned to symbolize 
# short-term (0–39 min), long-term (>40 min) respectively
LONG_TERM = 40
model.K_set = pyo.Set(within=pyo.PositiveIntegers)

# Parameter
# tc[i] : Time category for ith patients. (1 = short, 2 = long)
# Create tc for every patient in N_set
def tc_init(model):
    dict_aux = {}
    for i in model.N_set:
        # https://pyomo.readthedocs.io/en/stable/pyomo_modeling_components/Expressions.html
        # Because model elements result in expressions, not values, 
        # the following does not work as expected in an abstract mode:
        # if model.t[i] < LONG_TERM:
        if pyo.value(model.t[i]) < LONG_TERM:
            dict_aux[i] = 1
        else:
            dict_aux[i] = 2
    return dict_aux
        
model.tc = pyo.Param(model.N_set, initialize=tc_init)


# Set
# set_tc[k] : Set of patients indices with same value of k in n_set
def set_tc_k1_init(model):
    # set_aux = []
    for i in model.n_set:
        if pyo.value(model.tc[i]) == 1:
            yield i
            #set_aux.append(i)
    # return set_aux
#
def set_tc_k2_init(model):
    # set_aux = []
    for i in model.n_set:
        if pyo.value(model.tc[i]) == 2:
            yield i
            #set_aux.append(i)
    # return set_aux

# TODO : CREATE A SET (set_tc[]) WITH K_set DIMENSIONS
model.set_tc_1 = pyo.Set(initialize=set_tc_k1_init)
model.set_tc_2 = pyo.Set(initialize=set_tc_k2_init)


# VARIABLES -----------------------------------------------

# variables
# G[j] : Total physiotherapy time assigned to j-th physiotherapist
model.G = pyo.Var(model.S_set, within=pyo.NonNegativeIntegers)
#
# G_ave : Average physiotherapy time assigned to physiotherapists
model.G_ave = pyo.Var(within=pyo.NonNegativeReals)
#
# d_plus_range : set of d_plus variables in the objetive function
# TODO : NOT TO USE A NUMBER 4, USE A PARAMETER
model.d_plus_range = pyo.RangeSet(4)
# model.d_plus_range = pyo.Set(initialize=model.W_set[:-1])
#
# d_plus : Goals in objetive function
#   d_plus[1] : represents the absolute deviation 
#       from the goal of balanced time distribution among physiotherapists
#
#   d_plus[2] : absolute deviations 
#       from the goal of balanced distribution of patients 
#       in terms of time categories for short operations 
#
#   d_plus[3] : absolute deviations 
#       from the goal of balanced distribution of patients 
#       in terms of time categories for long operations
# 
#   d_plus[4] : TODO : GIVE AN EXPLANATION TO THIS VARIABLE. I DONT KNOW WHAT IT IS 
#       ?? is the deviation terms related to loading physiotherapists above their daily capacities?
model.d_plus = pyo.Var(model.d_plus_range, within=pyo.NonNegativeReals)
#
# d_minus_4 : is the deviation terms related to
# loading physiotherapists below their daily capacitie
model.d_minus_4 = pyo.Var(within=pyo.Reals)
#
# NP[j,k] : Number of patients assigned to j-th physiotherapist from k-th time category
model.NP = pyo.Var(model.S_set, model.K_set, within=pyo.NonNegativeIntegers)
#
# NP_ave[k] : Average number of patients assigned from k-th time category.
model.NP_ave = pyo.Var(model.K_set, within=pyo.NonNegativeReals)
#
#
# Decision variable
# y[i,j] : 
#   1 if i-th patient is assigned to j-th physiotherapist
#   0 otherwise
model.y = pyo.Var(model.n_set, model.S_set, within=pyo.Binary, initialize=0)

# --------------------------------------------------------
# --------------------------------------------------------

# OBJETIVE FUNCTION AND CONSTRICTIONS ---------------
# Maximize number of patients priority and time
def obj_rule(model):
    # return model.W[1] * model.d_plus[1] + model.W[2] * model.d_plus[2] + model.W[3] * model.d_plus[3] + model.W[4] * model.d_plus[4] + model.W[5] * model.d_minus_4
    return pyo.sum_product(model.W, model.d_plus, index=model.d_plus_range) + model.W[5] * model.d_minus_4

model.OBJ= pyo.Objective(rule=obj_rule, sense=pyo.minimize)

# Constraint. Eq (4): d_plus[1] : absolute time deviation
import pyomo.core.util as pyo_util

def d_plus_1_rule_ABS_OLD(model):
    # IT DOES NOT RUN
    import pyomo.core.expr.current as pyo_curr
    return model.d_plus[1] == pyo_util.quicksum(pyo_curr.AbsExpression(model.G[j] - model.G_ave) for j in model.S_set)

# Parameter
# M0 : Big M parameter, such that M0 > |G[j] - G_ave| , for all j in S_set
# We know: h >= G[j] >= |G[j] - G_ave| ==> h+1 > |G[j] - G_ave|
model.M0 = pyo.Param(initialize=model.h + 1)

# Variable
# Binary variable
model.B = pyo.Var(model.S_set, within=pyo.Binary, initialize=0)

# Constraint. Eq (13a)
def d_plus_1_rule_a(model, j):
    return (model.G[j] - model.G_ave) + model.M0 * model.B[j] >= 0

model.d_plus_1_constr_a = pyo.Constraint(model.S_set, rule=d_plus_1_rule_a)

# Constraint. Eq (13b)
def d_plus_1_rule_b(model, j):
    return -(model.G[j] - model.G_ave) + model.M0 * (1 - model.B[j]) >= 0

model.d_plus_1_constr_b = pyo.Constraint(model.S_set, rule=d_plus_1_rule_b)

# Constraint. Eq (13c)
def d_plus_1_rule_c(model):
    # d_plus_1 = 0
    # for j in model.S_set:
    #     d_plus_1 += (1 - 2 * model.B[j]) * (model.G[j] - model.G_ave)
    # return model.d_plus[1] == d_plus_1
    return model.d_plus[1] == pyo.quicksum((1 - 2 * model.B[j]) * (model.G[j] - model.G_ave) for j in model.S_set)

model.d_plus_1_constr_c = pyo.Constraint(rule=d_plus_1_rule_c)


# Constraint. Eq (5): G[j] : Total physiotherapy time assigned to jth physiotherapist
def Gj_constr_rule(model, j):
    return model.G[j] == pyo_util.quicksum(model.t[i] * model.y[i,j] for i in model.n_set)

model.Gj_constr = pyo.Constraint(model.S_set, rule=Gj_constr_rule)

# Constraint. Eq (6): G_ave : Average physiotherapy time assigned to physiotherapists
def G_ave_constr_rule(model):
    return model.G_ave == pyo_util.quicksum(model.G[j] for j in model.S_set)/model.S

model.G_ave_constr = pyo.Constraint(rule=G_ave_constr_rule)

# Constraint. Eq (7): d_plus[2], d_plus[3] : 
#   Absolute deviations from the goal of balanced
#   distribution of patients in terms of time categories,
#   respectively for short and long operations 
#
# Parameter
# M[K] : Big M parameter, such that M[K] > |N[j,K] - N_ave[K| , for all j in S_set, for all k in K_set
def M_init(model):
    return (model.c + 1)

model.M = pyo.Param(model.K_set, initialize=M_init)

# Variable
# Binary variable
model.C =pyo.Var(model.S_set, model.K_set, within=pyo.Binary, initialize=0)

# Constraint. Eq (14a)
def d_plus_23_rule_a(model, j, k):
    return (model.NP[j,k] - model.NP_ave[k]) + model.M[k] * model.C[j,k] >= 0

model.d_plus_23_constr_a = pyo.Constraint(model.S_set, model.K_set, rule=d_plus_23_rule_a)

# Constraint. Eq (14b)
def d_plus_23_rule_b(model, j, k):
    return -(model.NP[j,k] - model.NP_ave[k]) + model.M[k] * (1 - model.C[j,k]) >= 0

model.d_plus_23_constr_b = pyo.Constraint(model.S_set, model.K_set, rule=d_plus_23_rule_b)

# Constraint. Eq (14c)
def d_plus_23_rule_c(model, k):
    return model.d_plus[k] == pyo.quicksum((1 - 2 * model.C[j,k]) * (model.NP[j,k] - model.NP_ave[k]) for j in model.S_set)

model.d_plus_23_constr_c = pyo.Constraint(model.K_set, rule=d_plus_23_rule_c)


# Constraint. Eq (8a): NP[j,k] : 
# Number of patients assigned to j-th physiotherapist from k-th time category
def NP_rule(model, j, k):
    if pyo.value(k) == 1:
        return model.NP[j,k] == pyo_util.quicksum(model.y[i,j] for i in model.set_tc_1)
    elif pyo.value(k) == 2:
        return model.NP[j,k] == pyo_util.quicksum(model.y[i,j] for i in model.set_tc_2)
    else:
        return pyo.Constraint.Skip

model.NP_constr = pyo.Constraint(model.S_set, model.K_set, rule=NP_rule)

# Constraint. Eq (8b): NP[j,k] : 
# Maximum number of patients assigned to j-th physiotherapist
def NP_max_rule(model,j):
    return (0, pyo_util.quicksum(model.NP[j,k] for k in model.K_set), pyo.value(model.c))
model.NP_max_constr = pyo.Constraint(model.S_set, rule=NP_max_rule)

# Constraint. Eq (9): NP_ave[k] : 
# Average number of patients assigned from k-th time category.
def NP_ave_rule(model, k):
    return model.NP_ave[k] == pyo_util.quicksum(model.NP[j,k] for j in model.S_set)/model.S

model.NP_ave_costr = pyo.Constraint(model.K_set, rule=NP_ave_rule)

# Constraint Eq (10a) : d_plus[4]
# Physiotherapists loading daily capacities
def d_plus_4_rule(model):
    return model.d_plus[4] == pyo_util.quicksum(model.y[i,j] * model.t[i] for i in model.n_set for j in model.S_set)

model.d_plus_4_rule_constr = pyo.Constraint(rule=d_plus_4_rule)

# Constraint Eq (10b) : d_minus_4
# is the deviation terms related to loading physiotherapists below their daily capacities
def d_minus_4_rule(model):
    # return model.d_minus_4 == model.h * model.S - model.d_plus[4]
    return model.d_minus_4 == model.T - model.d_plus[4]

model.d_minus_4_rule_constr = pyo.Constraint(rule=d_minus_4_rule)


# Constraint (11) : one_i_one_j
# ensures that each patient is assigned to only one physiotherapist
def one_i_one_j_rule(model, i):
    return pyo_util.quicksum(model.y[i,j] for j in model.S_set) == 1

model.one_i_one_j_rule_constr = pyo.Constraint(model.n_set, rule=one_i_one_j_rule)

# Constraint (12) : daily_j_work
# Total physiotherapy time assigned to j-th physiotherapist less or equal than 
# daily work minutes per each physiotherapist
def daily_j_work_rule(model, j):
    # 0 <= model.G[j] <= model.h
    return (0, model.G[j], pyo.value(model.h))
    # return model.G[j] <= model.h

model.daily_j_work_constr = pyo.Constraint(model.S_set, rule=daily_j_work_rule)


# SOLVE ABSTRACT MODEL ------------------------------------

# Select a solver between: 'glpk', 'cplex'
SOLVER = 'glpk'
  
opt = pyo.SolverFactory(SOLVER)    

print("MODEL CONSTRUCTED = ", model.is_constructed())

# data = pyo.DataPortal()
# data.load(filename=ppshd_cfg.LM_STAGE_1_DAT_FILE, model=model)
# data.load(filename=ppshd_cfg.LM_STAGE_2_DAT_FILE, model=model)
# instance = model.create_instance(data)

instance = model.create_instance(ppshd_cfg.LM_STAGE_2_DAT_FILE)

print("INSTANCE CONSTRUCTED = ", instance.is_constructed())

# An attribute on an Abstract component cannot be accessed until the component 
# has been fully constructed (converted to a Concrete component)
# TODO : FIND A SOLVER THAT CAN SOLVE MINLP AND ERASE THIS CONDITION PLEASE!!!!! ---------------------------
if (SOLVER == 'glpk' or SOLVER == 'cplex'):
    # Deactivate non linear constraints
    instance.d_plus_1_constr_c.deactivate()
    instance.d_plus_23_constr_c.deactivate()


# To avoid automatic loading of the solution from the results object to the model, 
# use the load solutions=False argument to the call to solve().
results = opt.solve(instance, load_solutions=False)  # solves and updates instance

# @:tail


def solver_termination_info():
    """
    Create string with solver termination info
    """
    solver_info_str = "\ninstance.name = " + instance.name

    from pyomo.opt import SolverStatus, TerminationCondition
    solver_info_str += "\nsolver.status = " + str(results.solver.status)
    solver_info_str += "\nsolver.termination_condition = " + str(results.solver.termination_condition)

    if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
        solver_info_str += "\nWe have an optimal solution"
    else:
        solver_info_str += "\nSolve failed."
    return solver_info_str


def model_info():
    """
    Create string with model data information including parameters and sets
    """
    model_info_str = "\nN = " + str(pyo.value(instance.N))
    model_info_str += "\nn = " + str(pyo.value(instance.n))
    str_aux = "\nn_set = "
    for i in instance.n_set:
        str_aux += str(pyo.value(i)) + ", "
    model_info_str += str_aux
    
    model_info_str += "\nS = " + str(pyo.value(instance.S))
    str_aux = "\nS_set = "
    for i in instance.S_set:
        str_aux += str(pyo.value(i)) + ", "
    model_info_str += str_aux
    
    model_info_str += "\nh = " + str(pyo.value(instance.h))
    model_info_str += "\nc = " + str(pyo.value(instance.c))

    model_info_str += "\nM0 = " + str(pyo.value(instance.M0))

    str_aux = "\n"
    for i in instance.K_set:
        str_aux += "M["+str(i)+"] = " + str(pyo.value(instance.M[i])) + ", "
    model_info_str += str_aux

    str_aux = "\nW_set = "
    for i in instance.W_set:
        str_aux += str(pyo.value(i)) + ", "
    model_info_str += str_aux

    str_aux = "\nd_plus_range = "
    for i in instance.d_plus_range:
        str_aux += str(pyo.value(i)) + ", "
    model_info_str += str_aux

    str_aux = "\n"
    for i in instance.W_set:
        str_aux += "W["+str(i)+"] = " + str(pyo.value(instance.W[i])) + ", "
    model_info_str += str_aux
    
    str_aux = "\nK_set = "
    for i in instance.K_set:
        str_aux += str(pyo.value(i)) + ", "
    model_info_str += str_aux
    
    str_aux = "\n"
    for i in instance.n_set:
        str_aux += "tc["+str(i)+"] = " + str(pyo.value(instance.tc[i])) + ", "
    model_info_str += str_aux

    str_aux = "\nset_tc_1 = "
    for i in instance.set_tc_1:
        str_aux += str(pyo.value(i)) + ", "
    model_info_str += str_aux

    str_aux = "\nset_tc_2 = "
    for i in instance.set_tc_2:
        str_aux += str(pyo.value(i)) + ", "
    model_info_str += str_aux
    
    return model_info_str


def print_constraints():
    instance.OBJ.pprint()
    instance.d_plus_1_constr_a.pprint()
    instance.d_plus_1_constr_b.pprint()
    instance.d_plus_1_constr_c.pprint()
    instance.Gj_constr.pprint()
    instance.G_ave_constr.pprint()
    instance.d_plus_23_constr_a.pprint()
    instance.d_plus_23_constr_b.pprint()
    instance.d_plus_23_constr_c.pprint()
    instance.NP_constr.pprint()
    instance.NP_max_constr.pprint()
    instance.NP_ave_costr.pprint()
    instance.d_plus_4_rule_constr.pprint()
    instance.d_minus_4_rule_constr.pprint()
    instance.one_i_one_j_rule_constr.pprint()
    instance.daily_j_work_constr.pprint()

def print_results():
    print(instance.name)

    print("G_ave = ", pyo.value(instance.G_ave))
    for k in instance.K_set:
        print("NP_ave[" + str(k) +"] = ", pyo.value(instance.NP_ave[k]))
    
    str_aux = ""
    for j in instance.S_set:
        str_aux = "\nPhisioterapist " + str(j) + ". Patients : "
        total_time_j = 0
        short_time_j = 0
        long_time_j = 0
        patients_tc1 = ""
        patients_tc2 = ""
        # for i in instance.n_set:
        for i in (instance.set_tc_1):
            if pyo.value(instance.y[i,j]) != 0:
                str_aux += str(i) + ", "
                total_time_j += instance.t[i]
                short_time_j += instance.t[i]
                patients_tc1 += str(i) + ", "
        for i in (instance.set_tc_2):
            if pyo.value(instance.y[i,j]) != 0:
                str_aux += str(i) + ", "
                total_time_j += instance.t[i]
                long_time_j += instance.t[i]
                patients_tc2 += str(i) + ", "
        
        print(str_aux)

        for k in instance.K_set:
            print("NP[" + str(j) + "," + str(k) + "] = ", pyo.value(instance.NP[j,k]))
            print("C[" + str(j) + "," + str(k) + "] = ", pyo.value(instance.C[j,k]))

        print("patients_tc1 =", patients_tc1)
        print("patients_tc2 =", patients_tc2)
        print("total_time_j[" + str(j) + "] = ", str(total_time_j))
        print("short_time_j[" + str(j) + "] = ", str(short_time_j))
        print("long_time_j[" + str(j) + "] = ", str(long_time_j))
        print("G[" + str(j) + "] = ", pyo.value(instance.G[j]))
        print("B[" + str(j) + "] = ", pyo.value(instance.B[j]))
        

    print("RESULTS WITH SOLVER --------------------")
    print("model.OBJ =", pyo.value(instance.OBJ))
    

def print_paper_results():
    print("RESULTS IN PAPER -----------------------")
    patients_paper_indices = [1, 2, 3, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 28, 29, 30, 34, 35, 38, 41, 42, 47, 48, 49, 50, 51, 53, 54, 56, 60, 61, 62, 63, 64, 65, 66, 69, 71, 72, 74, 75, 76, 79, 80, 85, 86, 87]
    # patients for j phisioterapist:
    patients_paper = [(75,35,52,41,56,1,62,19,49,20,6,13,21,30), \
                (14,71,38,12,48,63,53,47,11,29,87,34,26,9), \
                    (61,25,15,69,27,50,64,16,54,42,86,10,18), \
                        (80,3,17,85,76,79,51,60,28,65,74,2,66)]
    # d_plus_paper
    d_plus_paper = [None] * pyo.value(4)
    # d_plus_paper
    # G_paper : Total physiotherapy time assigned to j-th physiotherapist.
    print("S =", str(pyo.value(instance.S)))
    G_paper = [None] * pyo.value(instance.S)
    for j in instance.S_set:
        G_paper[j-1] = sum(instance.t[i] for i in patients_paper[j-1])
        print("G_paper["+ str(j) +"]", str(G_paper[j-1]))
    
    # G_ave_paper : Average physiotherapy time assigned to physiotherapists.
    for j in instance.S_set:
        G_ave_paper = sum(G_paper[j-1] for j in instance.S_set)/pyo.value(instance.S)
    print("G_ave_paper =", str(G_ave_paper))

    # d_plus_paper[1]
    d_plus_paper[1-1] = sum(abs(G_paper[j-1] - G_ave_paper) for j in instance.S_set)
    print("d_plus_paper[1] =", d_plus_paper[1-1])

    # NP_paper[j,k] : Number of patients_paper assigned to j-th physiotherapist from k-th time category
    import numpy as np
    NP_paper = np.zeros(shape=(pyo.value(instance.S), len(instance.K_set)))
    for j in instance.S_set:
        print("PAPER. phisioterapist", j)
        for i in patients_paper[j-1]:
            if instance.t[i] < LONG_TERM:
                NP_paper[j-1, 1-1] +=1
            else:
                NP_paper[j-1, 2-1] +=1
                # print("instance.t["+str(i)+"] =", str(instance.t[i]))
        print("NP_paper["+str(j)+",1] =", str(NP_paper[j-1, 1-1]))
        print("NP_paper["+str(j)+",2] =", str(NP_paper[j-1, 2-1]))
    
    # NP_ave_paper[k] : Average number of patients_paper assigned from k-th time category
    NP_ave_paper = np.zeros(shape=(len(instance.K_set)))
    for k in instance.K_set:
        NP_ave_paper[k-1] = sum(NP_paper[j-1,k-1] for j in instance.S_set)/pyo.value(instance.S)
        print("NP_ave_paper["+str(k)+"] =", str(NP_ave_paper[k-1]))

    # d_plus_paper[2], d_plus_paper[3]
    for k in instance.K_set:
        d_plus_paper[k+1-1] = sum(abs(NP_paper[j-1, k-1] - NP_ave_paper[k-1]) for j in instance.S_set)
        print("d_plus_paper["+str(k+1)+"] =", str(d_plus_paper[k+1-1]))
    
    # d_plus_paper[4]
    d_plus_paper[4-1] = 0
    for j in instance.S_set:
        for i in patients_paper[j-1]:
            d_plus_paper[4-1] += instance.t[i]
    print("d_plus_paper[4] =", d_plus_paper[4-1])

    # d_minus_4_paper
    d_minus_4_paper = instance.h * instance.S - d_plus_paper[4-1]
    print("d_minus_4_paper =", d_minus_4_paper)


    # Calculate paper objetive
    print("Calculate paper objetive")
    paper_obj = instance.W[5] * d_minus_4_paper

    print("paper_obj =", str(paper_obj))
    for j in instance.S_set:
        print("d_plus_paper["+str(j)+"] =", str(d_plus_paper[j-1]))

    for i in instance.S_set:
        paper_obj += sum(instance.W[j] * d_plus_paper[j-1] for j in instance.S_set)
    print("PAPER.OBJ =", str(paper_obj))



print(model_info())

print_constraints()

print(solver_termination_info())

from pyomo.opt import SolverStatus, TerminationCondition
# To avoid automatic loading of the solution from the results object to the model, 
# use the load solutions=False argument to the call to solve().
if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
    # Manually load the solution into the model
    instance.solutions.load_from(results)
    print_results()
    # print_paper_results()

