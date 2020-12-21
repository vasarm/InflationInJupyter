import numpy as np
import sympy as sp

import matplotlib.pyplot as plt

import modules.errors as errors


def _create_parameter_key(parameter_combination):
    temp_list = []
    for key, value in sorted(parameter_combination.items(), key=lambda x: str(x[0])):
        temp_list.append(str(key) + "=" + str(value))
    return ", ".join(temp_list)

def plot_scalar_field_end_values(functions, parameter_combination, end_values, scalar_field_range):
    """
    Plots graph of scalar field end values on two graphs:
        1) φ - V(φ) (potential)
        2) φ - ε(φ) (epsilon, which defines infltaion end as ε=1)
    These graphs help user to select right value if there is a case, where multiple end values are found.

    Parameters
    ----------
    functions : dictionary
        Dictionary of two functions (V and ε).
    parameter_combination : dict
        Combination of parameter values
    end_values : list[float]
        List of scalar field end values
    scalar_field_range : array_like
        Scalar field interval which is defined in settings.

    Returns
    -------
    figure
        Returns figure of created plot.
    """
    def plot_range(x_points, x_min, x_max):
        """Calculate reasonable plot ranges.

        Returns
        -------
        list[float]
            Returns plotting range start and end value.
        """
        mean = np.sum(np.absolute(x_points)) / len(x_points)
        x_low = max([x_min, x_points.min() - mean])
        x_high = min([x_max, x_points.max() + mean])
        return [x_low, x_high]

    with np.errstate(divide="ignore", invalid="ignore"):
        x_scalar_field = scalar_field_range

        y_potential = functions["V"](x_scalar_field, **parameter_combination)
        y_potential_at_inflation_end = functions["V"](end_values, **parameter_combination)

        y_epsilon = functions["e"](x_scalar_field, **parameter_combination)
        y_epsilon_at_inflation_end = functions["e"](end_values, **parameter_combination)

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

        # Potential graph design
        ax1.set_title("$V(\\phi)$ graph")
        ax1.set_xlabel("$\\phi$")
        ax1.set_ylabel("$V(\\phi)$")
        ax1.set_xlim(plot_range(end_values, x_scalar_field.min(), x_scalar_field.max()))
        # ylim value is defined by potential values where φ = possible scalar field value
        # in the end of inflation
        ax1.set_ylim(plot_range(y_potential_at_inflation_end, y_potential.min(), y_potential.max()))
        ax1.grid(True)

        # Epsilon graph design
        ax2.set_title("$\\epsilon(\phi)$ graph")
        ax2.set_xlabel("$\\phi$")
        ax2.set_ylabel("$\\epsilon(\\phi)$")
        ax2.set_xlim(plot_range(end_values, x_scalar_field.min(), x_scalar_field.max()))
        ax2.set_ylim([-1, 4])
        ax2.grid(True)
        #numerate_points = [str(num) for num in range(1, len(end_values) + 1)]
        style = dict(size=10, color='red', weight='bold', textcoords='offset pixels', xytext=(0, 6), ha='center',va='bottom' )
        for num, point_x in enumerate(end_values):
            ax1.annotate(str(num+1), (point_x, y_potential_at_inflation_end[num]), **style)
            ax2.annotate(str(num+1), (point_x, y_epsilon_at_inflation_end[num]), **style)
        fig.subplots_adjust(hspace=0.6)
        ax1.plot(x_scalar_field, y_potential)
        ax1.scatter(end_values, y_potential_at_inflation_end)
        ax2.plot(x_scalar_field, y_epsilon)
        ax2.scatter(end_values, y_epsilon_at_inflation_end)

    return fig


def ask_scalar_field_end_value(end_values, ask_user_defined_point=False):
    """
    Function which asks user which scalar field value to choose as end point.
    This is runned when there are more than 1 solution.
    In case of numerical calculation user can add own it's own value because numerical
    method can be a little bit inaccurate (depends on used method).
    This method runs in infinite while cycle till user inputs right value.

    Parameters
    ----------
    end_values : list[float]
        List of possible scalar field values in the end of inflation.
    ask_user_defined_point : bool, optional
        If calculation was done numerically, by default False
    """

    if ask_user_defined_point:
        print("0. Select your own value.\n...   ")
    for num, elem in enumerate(end_values):
        print("Point " + str(num + 1) + ". φ = " + str(elem))

    def ask_cycle():
        try:
            user_input = int(input("What point to use (enter number): "))
            if user_input == 0 and ask_user_defined_point:
                while True:
                    try:
                        user_input_2 = np.float(input("Enter approximate value (write 'back' to go back)"))
                        if user_input_2.strip() == "back":
                            break
                        return user_input_2
                    except:
                        print("Couldnt convert it to number.")
            return end_values[user_input - 1]
        except:
            print("There was an error.")
            return ask_cycle()

    return ask_cycle()


def plot_N_function_graphs(functions, N_range):
    """If there exists more then one symbolic solutions for φ_0(N) [Scalar field initial value dependeci on inflation scale] functions then this function is called.
    It plots all possible solutions to help select the right solution.

    Parameters
    ----------
    functions : list[sympy function]
        N functions which are plotted
    N_range : np.array
        List of all N values.

    Returns
    -------
    figure
        Figure of created graph.
    """

    N_symbol = sp.Symbol("N", real=True, positive=True)
    function_names = ["{}. Function: {}".format(str(num), str(name)) for num, name in enumerate(functions, 1)]
    functions_numpy = [sp.lambdify(N_symbol, function, "numpy") for function in functions]

    fig, ax = plt.subplot()
    ax.set_title("$N(\\phi) \\text{ plot}$")
    ax.set_xlabel("$\phi_0$")
    ax.set_ylabel("$\phi_0(N)$")

    for num, name in enumerate(function_names):
        ax.plot(N_range, functions_numpy[num](N_range), label=name)

    return fig

def ask_right_N_function(functions):
    """
    This is infinite function till user chooses valid function.

    Parameters
    ----------
    functions : list
        List of possible N functions.

    Returns
    -------
    sp.Expr
        Returns sympy function what user chose.
    """
    for num, name in enumerate(functions, 1):
        print("{}. Function: {}".format(num, name))

    def ask_cycle():
        try:
            user_input = int(input("What function to use (enter number): "))
            return functions[user_input - 1]
        except:
            print("There was an error.")
            return ask_cycle()

    return ask_cycle()

def plot(plot_type, functions_dict, parameter_combinations, N_points, N_domain,
         scalar_field_domain, scalar_field_end_values, plot_id, model_name, info):
    """
    Function calls functions which plot graphs. Plot is decided by plot_type value.

    Parameters
    ----------
    plot_type : int
        1 -> n_s - r graph
        2 -> N - φ_0(N) graph
        3 -> φ_end - epsilon graph
    functions_dict : dict
        Contains all required functions for plotting.
    parameter_combinations : list[dictionary]
        All parameter combinations
    N_points : list
        Which N values are marked as points
    N_domain : array_like
        N values defined in settings.
    scalar_field_domain : array_like
        Domain of scalar field. Defined in config.Settings
    scalar_field_end_values : dictionary
        Scalar field end values for all combinations
    model_name : string
        Current model name.
    info : bool
        Boolean to show all points in ns-r graph (graph_type=1).
    """
    if plot_type not in [1, 2, 3]:
        raise errors.IncorrectValueError("Plot type must be 1, 2 or 3.")
    
    with np.errstate(divide="ignore", invalid="ignore"):
        if plot_type == 1:
            if N_points is None:
                raise errors.WrongTypeError("N values not defined.")
            plot1(functions_dict, parameter_combinations, N_points, N_domain, plot_id, model_name, info)

        elif plot_type == 2:
            if N_points is None:
                raise errors.WrongTypeError("N values not defined.")
            plot2(functions_dict, parameter_combinations, N_domain, plot_id, model_name)


        elif plot_type == 3:
            if scalar_field_domain is None:
                raise errors.WrongTypeError("Scalar field domain not defined.")
            elif scalar_field_end_values is None:
                raise errors.WrongTypeError("End value dictionary not defined.")
            plot3(functions_dict, parameter_combinations, scalar_field_domain, scalar_field_end_values, plot_id, model_name)

        plt.legend(bbox_to_anchor=(1.01, 1.01), loc="upper left")
        plt.tight_layout()

def plot2(functions_dict, parameter_combinations, N_domain, plot_id, model_name):
    if plt.fignum_exists(102):
        plt.figure(102)
    elif plt.fignum_exists(plot_id):
        plt.figure(plot_id)
    else:
        if plot_id is not None:
            plt.figure(plot_id, figsize=(18, 12))
        else:
            plt.figure(figsize=(18, 12))
        plt.rcParams.update({'font.size': 20})
        plt.xlim([0, 100])
        plt.xlabel("$N$")
        plt.ylabel("$\\phi_{0}$$")
        plt.grid(True)

    N_values = N_domain[np.where(N_domain <= 100.0)]

    for param in parameter_combinations:
        key = _create_parameter_key(param)
        N_function = functions_dict["N"][key]
        if model_name:
            label_name = "{}: {}".format(model_name, key)
        else:
            label_name = "{}: {}".format("Param", key)
        plt.plot(N_values,
                 N_function(N_values),
                 label=label_name)


def plot3(functions_dict, parameter_combinations, scalar_field_domain, scalar_field_end_values, plot_id, model_name):
    if plt.fignum_exists(103):
        plt.figure(103)
    elif plt.fignum_exists(plot_id):
        plt.figure(plot_id)
    else:
        plt.rcParams.update({'font.size': 20})
        if plot_id is not None:
            plt.figure(plot_id, figsize=(18, 12))
        else:
            plt.figure(figsize=(18, 12))
        plt.xlabel("$\\phi$")
        plt.ylabel("$\\epsilon(\\phi)$")
        plt.ylim([-1, 2])
        plt.grid(True)

    for param in parameter_combinations:
        key = _create_parameter_key(param)
        if model_name:
            label_name = "{}: {}".format(model_name, key)
        else:
            label_name = "{}: {}".format("Params.", key)

        domain = domain[np.where(domain >= scalar_field_end_values[key])]
        epsilon_values = functions_dict["e"](domain, **param).reshape((-1))

        plt.plot(domain, epsilon_values,
                 label=label_name, alpha=0.6)

def plot1(functions, parameter_combinations, N_values, N_domain, plot_id, model_name, info):
    """Function for plotting n_s - r graph.
    Parameters
    ----------
    functions : dict[]
        Dictionary of needed functions
    """

    """
    Calculate n_s and r values for defined N value.
    Key = N_value and it's value is x and y coordinates in n_s and r graph.
    """
    N_points = {}

    # Default settings for N value markers
    color = "k"  # "k" means black
    markers = {1: "o" + color, 2: "P" + color, 3: "s" + color,
               4: "*" + color, 5: "8" + color, 6: "x" + color}

    if plt.fignum_exists(101):
        plt.figure(101)
    elif plt.fignum_exists(plot_id):
        plt.figure(plot_id)
    else:
        if plot_id is not None:
            plt.figure(plot_id, figsize=(18, 12))
        else:
            plt.figure(figsize=(18, 12))
        plt.rcParams.update({'font.size': 20})
        plt.axis([0.95, 0.98, 0, 0.14])
        plt.xlabel("$n_s$")
        plt.ylabel("$r$")
        plt.grid(True)

        # Plot Planck satellite data
        planck = np.genfromtxt("data.csv", delimiter=",")
        planck = list(zip(*planck))

        Planck_alpha = 0.3  # alpha value for Planck's data
        plt.fill_between(planck[0], planck[1], alpha=Planck_alpha, label="TT,TE,EE+lowE")
        plt.fill_between(planck[2], planck[3], alpha=Planck_alpha, label="TT,TE,EE+lowE+lensing")
        plt.fill_between(planck[4], planck[5], alpha=Planck_alpha, label="+BK15+BAO")

    for param_combination in parameter_combinations:
        key = _create_parameter_key(param_combination)
        N_function = functions["N"][key]
        if model_name:
            label_name = "{}: {}".format(model_name, key)
        else:
            label_name = "{}: {}".format("Param", key)

        domain = N_function(N_domain)
        n_s = functions["ns"](N_domain, **param_combination)
        r = functions["r"](N_domain, **param_combination)
        plt.plot(n_s, r, label=label_name, alpha=0.6)

        for N in N_values:
            if info:
                if model_name:
                    print("#{} | {} : N={}, n_s={}, r={}, φ={}".format(model_name, key, str(N),
                                                            functions["ns"](N_function(N), **param_combination),
                                                            functions["r"](N_function(N), **param_combination),
                                                            N_function(N)
                                                            )
                        )
                else:
                    print("{} : N={}, n_s={}, r={}, φ={}".format(key, str(N),
                                                            functions["ns"](N_function(N), **param_combination),
                                                            functions["r"](N_function(N), **param_combination),
                                                            N_function(N)
                                                            )
                        )
            # Idea is to calculate n_s and r values for wanted N values and add them to dictionary
            # Then it is easier to plot them with the same markers
            if str(N) in N_points:
                N_points[str(N)].append((functions["ns"](N_function(N), **param_combination),
                                        functions["r"](N_function(N), **param_combination)))
            else:
                N_points[str(N)] = [(functions["ns"](N_function(N), **param_combination),
                                    functions["r"](N_function(N), **param_combination))]

    # Plot n_s - r points with certain N value
    for num, N_value in enumerate(N_points, 1):
        # If legend already has a value with same label, remove the previous entrie's label
        # and add new label to legend
        for plot_line in plt.gca().get_lines():
            if plot_line.get_label() == str(N_value):
                plot_line.set_label("")
        plt.plot(*zip(*N_points[N_value]), markers[num], label=str(N_value), alpha=0.6, markersize=10)


