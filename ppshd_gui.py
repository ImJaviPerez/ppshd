# ppshd_lm_gui.py Graphical Use Interfa (GUI) para el Travel Salesman Problem (TSP) con las restricciones de Miller, Tucker y Zemlin

import ppshd_cfg

# ---------------------------------------------------------
# ---------- LEER FICHERO CONFIGURACION -------------------

def read_data():
    """
    It returns a DataPortal object containing every parameter and set of the ".dat" file. This file has a pseudo-AMPL format
    """
    import ppshd_cfg
    import pyomo.environ as pyo

    # Objec to save configuration dat
    data = pyo.DataPortal()
    # Object to save abstract model
    model_init = pyo.AbstractModel()

    # Parmeters and sets in configuration data
    model_init.N_CITIES = pyo.Param(within=pyo.PositiveIntegers)
    def check_n_cities_rule(model_init):
        return model_init.N_CITIES >= 3
    model_init.check_N_CITIES = pyo.BuildCheck(rule=check_n_cities_rule)

    # Arcos entre nodos
    model_init.N = pyo.RangeSet(model_init.N_CITIES)
    
    # C[i,j] = distancia del nodo i al j = Matriz del coste
    model_init.C = pyo.Param(model_init.N, model_init.N, within=pyo.NonNegativeReals, initialize=0)

    data.load(filename=ppshd_cfg.LM_DAT_FILE, param=model_init.N_CITIES)
    data.load(filename=ppshd_cfg.LM_DAT_FILE, param=model_init.C)
    return data
# Datos del fichero de configuracion del modelo lineal
data = read_data()

# Save data in a numpy array
num_of_cities_dat = data['N_CITIES']
import numpy as np
data_arr = np.zeros(shape=(ppshd_cfg.NUM_CITIES_MAX,ppshd_cfg.NUM_CITIES_MAX))
for i in range(num_of_cities_dat):
    for j in range(i+1, num_of_cities_dat):
        data_arr[i,j] = data['C'][i+1, j+1]


# ---------------------------------------------------------
# ---------- MOSTRAR FICHERO CONFIGURACION EN GUI ---------
import PySimpleGUI as sg

def gather_data(last_values, new_number_of_cities: int = num_of_cities_dat):
    """
    Save GUI input text values from PySimpleGUI to a numpy array: data_arr
    """
    # TODO. CHEQUEAR QUE LOS DATOS INTRODUCIDOS SON CORRECTOS
    for i in range(new_number_of_cities):
        for j in range(i+1, new_number_of_cities):
            data_arr[i,j] = last_values['row_'+str(i+1)+"_col_"+str(j+1)]
            # print("data_arr[", i, "," , j, "]", data_arr[i,j])


# Create a backup of the data file
def backup_dat_file(source_file):
    """
    Create a backup of the current .dat file with a time stamp prefix in a new directory
    """
    # backup del fichero de datos
    import datetime, time
    # This return the epoch timestamp    
    epochTime = time.time()
    # We generate the timestamp 
    timeStamp = datetime.datetime.fromtimestamp(epochTime).strftime('%Y-%m-%d-%H-%M')

    import os
    backup_path = os.path.join(ppshd_cfg.BACKUP_DIR, timeStamp+"_"+source_file)
    import shutil
    # copy file from source to destination
    shutil.copy(src=source_file, dst=backup_path)


def create_new_dat_file(new_number_of_cities=num_of_cities_dat):
    import ppshd_cfg
    """
    Create a new dat file using data_arr
    """
    fo = open(ppshd_cfg.LM_DAT_FILE, "wt")
    # initial comment
    fo.write("# "+ppshd_cfg.LM_DAT_FILE+" AMPL format\n\n")

    # Parameter in AMPL format
    fo.write("param N_CITIES := " + str(new_number_of_cities) + ";\n\n")

    # Parameter table in AMPL format
    def C_to_str():
        # Maximum number of characters to use: length of each element
        max_len = 0
        for i in range(new_number_of_cities):
            if abs(data_arr.max(axis=None)) > max_len:
                max_len = abs(data_arr.max(axis=None))
        import math
        max_len = int(math.log10(max_len)) + 2
        # max_len = math.ceil(math.log10(max_len)) + 1
        # max_len = int(np.log10(max_len)) + 2
        

        # First line
        C_to_str = "param C :"
        C_1st_line = " "
        for i in range(new_number_of_cities):
            C_1st_line += str(i+1).rjust(max_len)
        C_to_str += C_1st_line + " :=\n"
        # Next lines
        for i in range(new_number_of_cities):
            C_line = str(i+1).rjust(10)
            for j in range(new_number_of_cities):
                if i != j:
                    if i<j:
                        C_line += str(int(data_arr[i,j])).rjust(max_len)
                    else:
                        C_line += str(int(data_arr[j,i])).rjust(max_len)
                else:
                    C_line += ".".rjust(max_len)
            C_to_str += C_line  + "\n"
        C_to_str += ";\n"
        return C_to_str

    fo.write(C_to_str())
    fo.close()
    # At the end we will append the optimal solution to this file


# Run solver and get solution
def run_solver():
    """Run solver and get solution"""
    import ppshd_stage_1_lm
    # import pyomo.environ as pyo

    print(ppshd_stage_1_lm.instance.name, '\n')

    from pyomo.opt import SolverStatus, TerminationCondition
    # To avoid automatic loading of the solution from the results object to the model, use the load solutions=False argument to the call to solve().
    if (ppshd_stage_1_lm.results.solver.status == SolverStatus.ok) and (ppshd_stage_1_lm.results.solver.termination_condition == TerminationCondition.optimal):
        # Manually load the solution into the model
        print("Tenemos solucion optima")
        ppshd_stage_1_lm.instance.solutions.load_from(ppshd_stage_1_lm.results)
    else:
        print("Solve failed.")
    return ppshd_stage_1_lm


# Get x_values from solution
def get_x_values():
    import pyomo.environ as pyo
    import numpy as np

    x_values_tmp = np.ndarray(shape=(pyo.value(ppshd_stage_1_lm.instance.N_CITIES), pyo.value(ppshd_stage_1_lm.instance.N_CITIES)), dtype=bool)

    for i in ppshd_stage_1_lm.instance.N:
        for j in ppshd_stage_1_lm.instance.N:
            x_values_tmp[i-1,j-1] = pyo.value(ppshd_stage_1_lm.instance.x[i,j])
    return x_values_tmp

# Get u_values from solution
def get_u_values():
    import pyomo.environ as pyo
    import numpy as np

    u_values_tmp = np.empty(shape=pyo.value(ppshd_stage_1_lm.instance.N_CITIES), dtype=int)

    for i in ppshd_stage_1_lm.instance.N:
        u_values_tmp[i-1] = pyo.value(ppshd_stage_1_lm.instance.u[i])
    return u_values_tmp

# Calculate optimal solution
def calculate_optimal_solution():
    import pyomo.environ as pyo

    ppshd_stage_1_lm_cost_tmp = 0
    for i in np.argsort(u_values):
        for j in range(pyo.value(ppshd_stage_1_lm.instance.N_CITIES)):
            if x_values[i,j] != 0:
                # ppshd_stage_1_lm_cost_tmp += pyo.value(ppshd_stage_1_lm.instance.x[i+1,j+1]) * pyo.value(ppshd_stage_1_lm.instance.C[i+1,j+1])
                ppshd_stage_1_lm_cost_tmp += pyo.value(ppshd_stage_1_lm.instance.C[i+1,j+1])
    return ppshd_stage_1_lm_cost_tmp

# Show optimal solution
def get_str_solution(prepend = "# \t"):
    """
    Create string with optimal route
    """
    import pyomo.environ as pyo

    str_solution = "\n\n"+prepend + "Posicion\tOrigen -> Destino\tDistancia\n"
    for i in np.argsort(u_values):
        str_solution += prepend + "u["+str(i+1)+"]="+str(u_values[i])
        for j in range(pyo.value(ppshd_stage_1_lm.instance.N_CITIES)):
            if x_values[i,j] != 0:
                str_solution += "\t\t"+ str(i+1) + " -> "+str(j+1)+"\t" + str(pyo.value(ppshd_stage_1_lm.instance.C[i+1,j+1])) + "\n"
    return str_solution


def show_route():
    """
    Print optimal route
    """
    import pyomo.environ as pyo

    for i in np.argsort(u_values):
        for j in range(pyo.value(ppshd_stage_1_lm.instance.N_CITIES)):
            if x_values[i,j] != 0:
                print("u[", i+1,"] = ", pyo.value(ppshd_stage_1_lm.instance.u[i+1]), \
                    " ,, x(", i+1, ",", j+1, ") =", x_values[i,j], " :::", pyo.value(ppshd_stage_1_lm.instance.x[i+1,j+1]), \
                    " ,, C(", i+1, ",", j+1, ") =", pyo.value(ppshd_stage_1_lm.instance.C[i+1, j+1]))


def save_optimal_to_TSV_file():
    # TODO --------------------------------------------
    pass

def append_optimal_to_data_file(str_to_write):
    # Append solution to the current AMPL data file
    fo = open(ppshd_cfg.LM_DAT_FILE, "at")
    fo.write(str(str_to_write))
    fo.close()




# Define the window's contents

def create_layout():
    """
    Create GUI layot line by line
    """
    # TODO CREAR USER SETTINGS PARA DEFINIR FONT DE INPUT Y TEXT
    # TODO CREAR TABGROUP Y TABS PARA IR VIENDO DIFERENTES PANTALLAS DE CONFIGURACION Y RESULTADOS FINALES
    import ppshd_cfg

    sg.set_options(font=("Courier New", 12))

    # Number of cities combo box
    str_n_cities = "Número de ciudades:"
    combo_values = [str(i) for i in range(ppshd_cfg.NUM_CITIES_MIN, ppshd_cfg.NUM_CITIES_MAX+1)]
    layout =[[sg.Text(str_n_cities, font=("Arial", 12), size=(len(str_n_cities),None))] + \
            [sg.Combo(combo_values, key="combo_n_cities", size=(5,1), default_value=num_of_cities_dat, readonly=True)] +\
                [sg.Button('Cambiar', font=("Arial", 12))]]
    # Zero row with headers of the columns
    layout = layout, [[sg.Text("   ", key="row_0_col_0", size=(3,None))] + \
    [sg.Text(i, key="row_0_col_"+str(i), visible=(i <= num_of_cities_dat), size=(3,None), justification = "right") for i in range(1, ppshd_cfg.NUM_CITIES_MAX+1)],]
    # Next lines with input data
    layout_dict = {}
    # https://stackoverflow.com/questions/66653381/update-window-layout-with-buttons-in-row-with-pysimplegui
    # sg.pin is an element provided into a layout so that when it's made invisible and visible again, it will be in the correct place. Otherwise it will be placed at the end of its containing window/column
    
    for n in range(1, ppshd_cfg.NUM_CITIES_MAX+1):
        # First element of the row: number of city
        layout_dict[n] = [sg.Text(n, visible=(n <= num_of_cities_dat), key="row_"+str(n)+"_col_0", size=(3,None))]
        if n!=1:
            # Insert empty elements at the begining
            layout_dict[n] = layout_dict[n] + [sg.Text("___", visible=(n <= num_of_cities_dat), key="row_"+str(n)+"_col_"+str(k), size=(3,None)) for k in range(1, n)]
        # Elemento vacio + textos modificables
        layout_dict[n] = [layout_dict[n] + [sg.Text(" . ", visible=(n <= num_of_cities_dat), key="row_"+str(n)+"_col_"+str(n), size=(3,None))] + \
            [sg.Input(int(data_arr[n-1,j-1]), visible=(n <= num_of_cities_dat and j <= num_of_cities_dat) , key='row_'+str(n)+"_col_"+str(j), size=(3,None), justification = "right") for j in range(n+1,ppshd_cfg.NUM_CITIES_MAX+1)]]
        # Append new line to GUI
        layout = layout, layout_dict[n]
    # Botones finales
    layout = layout, [sg.Button('Ok', font=("Arial", 12)), sg.Button('Quit', font=("Arial", 12))]
    return layout

def change_layout_visibility(number_of_visible_cities: int):
    import ppshd_cfg

    for i in range(ppshd_cfg.NUM_CITIES_MAX+1):
        for j in range(ppshd_cfg.NUM_CITIES_MAX+1):
            window["row_"+str(i)+"_col_"+str(j)].Update(visible = i <= number_of_visible_cities and j <= number_of_visible_cities)


# Create the window
window = sg.Window("Travel Salesman Problem", create_layout())
# Display and interact with the Window using an Event Loop
while True:
    event, values = window.read()
    # See if user wants to quit or window was closed
    if event == sg.WINDOW_CLOSED or event == 'Quit':
        break
    # Data file saving
    if event == 'Ok':
        # Get data from GUI
        gather_data(last_values=values, new_number_of_cities=int(values["combo_n_cities"]))
        backup_dat_file(ppshd_cfg.LM_DAT_FILE)
        create_new_dat_file(int(values["combo_n_cities"]))
        # Run solver
        ppshd_stage_1_lm = run_solver()
        print('Status: ', ppshd_stage_1_lm.results.solver.status, '\n')
        # Get variables solution
        x_values = get_x_values()
        u_values = get_u_values()
        # Calculate optimal solution
        ppshd_stage_1_lm_cost = calculate_optimal_solution()
        print('\nCoste x*C = ', ppshd_stage_1_lm_cost)
        print(get_str_solution())
        # show_route()
        # Save optimal solution to current AMPL data file
        append_optimal_to_data_file("\n\n# La solucion optima es "+str(ppshd_stage_1_lm_cost))
        # Save whole solution to current AMPL data file
        append_optimal_to_data_file(get_str_solution())
        # Save optimal solution to a XLS file
        save_optimal_to_TSV_file()
        

        break
    # Change number of cities
    if event == 'Cambiar':
        # Change number of cities
        # Change layout elements visibility
        change_layout_visibility(int(values["combo_n_cities"]))
        # ESTAS AQUI    # ESTAS AQUI    # ESTAS AQUI    # ESTAS AQUI    
        # TODO PONER PYOSYMPLEGUI.pin() en los BOTONES PARA RECOLOCARLOS EN EL SITIO ADECUADO
        # ESTAS AQUI    # ESTAS AQUI    # ESTAS AQUI    # ESTAS AQUI    
        
# Finish up by removing from the screen
window.close()


# ---------------------------------------------------------
# ---------------------------------------------------------


print("FIN DE MOSTRAR")





# https://stackoverflow.com/questions/38700214/pyomo-access-solution-from-python-code
# import pyomoio as po
# c_df = po.get_entity(instance, 'c').unstack()
# print(c_df)

