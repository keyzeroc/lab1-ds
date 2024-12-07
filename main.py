import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


# Модель генерації випадкової величини (похибки вимірювання)
def generate_exponential_error(lambda_param, size):
    return np.random.exponential(1 / lambda_param, size)


# Модель зміни досліджуваного процесу
def constant_process(value, size):
    return np.full(size, value)


# Адитивна модель експериментальних даних
def generate_experimental_data(process, errors):
    return process + errors


# Метод Монте-Карло
def monte_carlo_simulation(
    process_func, error_func, process_param, error_param, num_simulations, size
):
    simulations = []
    for _ in range(num_simulations):
        process = process_func(process_param, size)
        errors = error_func(error_param, size)
        experimental_data = generate_experimental_data(process, errors)
        simulations.append(experimental_data)
    return simulations


# Визначення статистичних характеристик
def calculate_statistics(data):
    mean = np.mean(data)
    variance = np.var(data)
    std_dev = np.std(data)
    return mean, variance, std_dev


# Дослідження зміни статистичних характеристик
def investigate_error_impact(
    process_func,
    error_func,
    process_param,
    initial_error_param,
    error_param_range,
    num_simulations,
    size,
):
    results = []
    for error_param in error_param_range:
        simulations = monte_carlo_simulation(
            process_func, error_func, process_param, error_param, num_simulations, size
        )
        all_means = [calculate_statistics(sim)[0] for sim in simulations]
        all_variances = [calculate_statistics(sim)[1] for sim in simulations]
        all_std_devs = [calculate_statistics(sim)[2] for sim in simulations]
        results.append(
            {
                "error_param": error_param,
                "mean": np.mean(all_means),
                "variance": np.mean(all_variances),
                "std_dev": np.mean(all_std_devs),
            }
        )
    return results


# Виведення результатів у консоль
def print_results(results):
    table = []
    for result in results:
        table.append(
            [
                result["error_param"],
                result["mean"],
                result["variance"],
                result["std_dev"],
            ]
        )
    print(
        tabulate(
            table,
            headers=["Error Param", "Mean", "Variance", "Std Dev"],
            tablefmt="pretty",
        )
    )


# Візуалізація результатів
def plot_results(results):
    error_params = [result["error_param"] for result in results]
    means = [result["mean"] for result in results]
    variances = [result["variance"] for result in results]
    std_devs = [result["std_dev"] for result in results]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.plot(error_params, means, marker="o")
    plt.title("Mean vs Error Parameter")
    plt.xlabel("Error Parameter")
    plt.ylabel("Mean")

    plt.subplot(1, 3, 2)
    plt.plot(error_params, variances, marker="o")
    plt.title("Variance vs Error Parameter")
    plt.xlabel("Error Parameter")
    plt.ylabel("Variance")

    plt.subplot(1, 3, 3)
    plt.plot(error_params, std_devs, marker="o")
    plt.title("Standard Deviation vs Error Parameter")
    plt.xlabel("Error Parameter")
    plt.ylabel("Standard Deviation")

    plt.tight_layout()
    plt.show()


# Виконання дослідження
error_param_range = np.linspace(0.1, 1, 10)
results = investigate_error_impact(
    constant_process, generate_exponential_error, 10, 0.5, error_param_range, 100, 1000
)

# Виведення та візуалізація результатів
print_results(results)
plot_results(results)
