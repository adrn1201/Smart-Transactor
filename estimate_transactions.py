from pyfiglet import figlet_format
from termcolor import colored
import numpy as np
import joblib


predictors = []

category_headers = (
    'Entertainment',
    'Food Dining',
    'Gas Transport',
    'Grocery Net Amount',
    'Grocery Point of Sale(POS)',
    'Health and Fitness',
    'Home',
    'Kids and Pets',
    'Music Net Amount',
    'Music Point of Sale(POS)',
    'Personal Care',
    'Shopping Net Amount',
    'Shopping Point of Sale(POS)',
    'Travel'
)

amount_msgs = (
    'Amount spent on Entertainment: ',
    'Amount spent on Food Dining: ',
    'Amount spent on Gas Transport: ',
    'Net Amount spent on Grocery: ',
    'Grocery(POS) amount: ',
    'Amount spent on health and fitness: ',
    'Amount spent on Home: ',
    'Amount spent on Kids and Pets: ',
    'Net Amount spent on Music: ',
    'Music(POS) amount: ',
    'Amount spent on Personal Care: ',
    'Net Amount spent on Shopping: ',
    'Shopping(POS) amount: ',
    'Amount spent on travel: '
)

frequency_msgs = (
    'How many times does he/she spend on entertainment: ',
    'How many times does he/she spend on Food Dining: ',
    'How many times does he/she spend on Gas Transport: ',
    'How many times does he/she spend on Grocery (Net): ',
    'How many times does he/she spend on Grocery (POS): ',
    'How many times does he/she spend on Health and Fitness: ',
    'How many times does he/she spend on home: ',
    'How many times does he/she spend on Kids and Pets: ',
    'How many times does he/she spend on Music(Net): ',
    'How many times does he/she spend on Music(POS): ',
    'How many times does he/she spend on Personal Care: ',
    'How many times does he/she spend on Shopping(Net): ',
    'How many times does he/she spend on Shopping(POS): ',
    'How many times does he/she spend on travel: '
)


def display_welcome_msg():
    header = figlet_format("\t\tWelcome to Smart Transactor")
    header = colored(header, color="yellow")
    print(header)
    sub_header = colored(
        'Predict the Sum of the Customer\'s using Ensemble Gradient Boosting Machine Learning Algorithm',
        color='cyan')
    print(f'\t\t\t{sub_header}')
    print('\n')


def display_options():
    print('Here are your options: \n')
    print(f'{colored("[1]", color="green")} - Start estimating')
    print(f'{colored("[2]", color="red")} - Quit')
    print('\n')


def print_dashes():
    print('-' * 70)


def display_category_header(header):
    return colored(f'{header} (Half of the Month Expense)\n', color='cyan')


def ask_user(amount_msg, freq_msg):
    amount = float(input(amount_msg))
    frequency = int(input(freq_msg))
    return predictors.extend([amount, frequency])


def load_regressor_model():
    return joblib.load('v1_trained_transaction_regressor_model.pkl')


def prompt_predicted_values():
    print_dashes()
    print(display_category_header('Half of the Month total Transaction'))
    hom_total_input = float(
        input('What is the current Half of the Month total transaction: '))
    predictors.append(hom_total_input)

    for i in range(14):
        print_dashes()
        print(display_category_header(category_headers[i]))
        ask_user(amount_msgs[i], frequency_msgs[i])


def predict_values():
    model = load_regressor_model()
    transaction_to_estimate = np.array([np.array(predictors)])
    predicted_value = model.predict(transaction_to_estimate)
    print_dashes()
    print(
        colored(
            'Customer\'s Sum of Transaction for the next Half of the Month: \n',
            color='green'))
    print(f'₱{colored(round(predicted_value[0], 2), color = "green")}')
    print_dashes()


def display_err_msg(msg):
    return colored(f'{msg}\n', color='red')


display_welcome_msg()


while True:
    try:
        display_options()
        choice = int(input('Please select an option: '))
        if choice == 1:
            prompt_predicted_values()
            predict_values()
            predictors.clear()
            print(
                colored(
                    'Do you want to predict the next customer\'s transaction again?\n',
                    color='cyan'))
        else:
            break
    except TypeError:
        print(display_err_msg('Please enter a valid number!'))
    except ValueError:
         print(display_err_msg('Please enter a valid value!'))


print(f'\n{colored("Thank you!", color = "magenta")}')
