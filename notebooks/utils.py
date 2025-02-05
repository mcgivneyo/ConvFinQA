# utils.py
"""utility functions for the data processing and model evaluation."""

import pandas as pd
import numpy as np
from pprint import pprint
from numerizer import numerize
from collections import Counter
from pathlib import Path
import math


def compare_value(val1, val2, pct_precision=1, num_precision=1, verbose=False)->int:
    """
    Compares values in each row of the table and returns 1 if they are the same, 0 otherwise.
    Handles both percentage formats and numeric formats.
    Compare values considering floating-point precision

    Args:
        val1 (str): A value from the first column.
        val2 (str): A value from the second column.
        pct_precision (int): The percentage precision.
        num_precision (int): The numeric precision.
        verbose (bool): Whether to print the values that are not the same.

    Returns:
        int: 1 if the values are the same, 0 otherwise.
        
    """
    score = 0
    if '%' in str(val1) and '%' not in str(val2):
        val1 = float(val1.rstrip('%'))
        val2 = float(f'{val2:.4f}')
        val2 = np.round(float(val2) * 100,1)
        score = 1 if abs(float(val1) - float(val2)) < pct_precision else 0

    else:
        val1 = float(val1.strip())
        val2 = np.round(float(val2), 0)
        score = 1 if abs(float(val1) - float(val2)) < num_precision else 0
    return score


def get_difficulty(annotation:dict)->str:
    """
    Get the relative difficulty of the question based on the simple/hybrid conversation type
    and the number of steps needed to solve the question.
        annotation (dict): The annotation dictionary containing question details.
        str: A string representing the difficulty level of the question.
    
        """
    return dict(Counter(annotation.get('qa_split', [])))


def get_q(row:pd.Series)->dict:
    """
    get the top level ( not decomposed) questions from the qa, qa_0, qa_1 columns

    """
    t = []
    for c in ['qa', 'qa_0', 'qa_1']:
        if isinstance(row[c], dict):
            t.append(row[c].get('question'))
    return {f'question_{i}':q for i,q in enumerate(t)}


def get_a(row:pd.Series)->dict:

    """
    get the answers and the execution results from the qa, qa_0, qa_1 columns

    """
    d = {}
    i = 0
    for c in ['qa', 'qa_0', 'qa_1']:
        if isinstance(row[c], dict):
            d[f'answer_{i}'] = row[c].get('answer')
            d[f'exe_ans_{i}'] = row[c].get('exe_ans')
            i += 1
    return d



def get_score(value_type: str, amt: float, gt_amt: float,
              num_tolerance: float = 1, pct_tolerance: float = 0.1) -> int:
    """
    compare amounts with a tolerance
    args:
    value_type: str: type of value (PCT or NET)
    amt: float: amount to compare
    gt_amt: float: ground truth amount
    num_precision: float: precision for numbers
    pct_precision: float: precision for percentages

    return:
    int: 1 if the amounts are equal (within tolerance), 0 otherwise

    """
    tolerance = num_tolerance
    if value_type=='PERCENT':
        if amt >1  and gt_amt < 1:
            amt /= 100
            tolerance = pct_tolerance

    return 1 if math.isclose(amt, gt_amt, rel_tol=tolerance) else 0


def score_answers(response:list, exe_ans_list:list, step_list: list)->dict:  # todo
    """
    Evaluate the results of the model
    args:
    response: list: list of answers
    exe_ans_list: list: list of expected answers
    step_list: list: list of expected steps for the program

    return:
    dict: scores

    'exe_ans_score': % of steps with correct answers
    'retrieval_score':  % correct retrieval steps (retrieval  = find in text, not an operation)
    'step_score': % of correct program steps 
    'exe_ans': final answer of the chain is correct
    """
    # some hacky thresholds for comparing numbers
    # num_precision = 1   
    # pct_precision = 0.1

    if len(response) != len(exe_ans_list):
        # todo: mismatch between the number of answers - create a better matching algorithm
        print('mismatch between the number of answers')
        print(response)
        print(exe_ans_list) 

    # create a dataframe to store the results, concatenate LLM results with the ground truth
    answer = pd.DataFrame([{'id': a.id, 'type': a.amount_type.value, 'operation':a.operation.__str__(), 'amount': a.amount} for a in response])
    gt_answer = pd.DataFrame([{'step_list': sl, 'exe_ans_list':  ex_a} for  ex_a, sl in zip(exe_ans_list, step_list)])
    answer = pd.concat([answer, gt_answer], axis=1)

    # clean up the operation column so it matches 
    answer.operation = answer.operation.map(lambda x: x.replace(', None)', '').replace('(', ' ') if x.startswith('Ask') else x)
    
    answer['score'] = answer.apply(lambda x: get_score(x.type, x.amount, x.exe_ans_list), axis=1)
    answer['retrieved_value'] = answer.operation.map(lambda x: 'Ask for number' in str(x))
    answer['step_score'] = answer.apply(lambda x: 1 if x.operation == x.step_list else 0, axis=1)
    answer['arg_num'] = answer.retrieved_value.apply(lambda x: None if x else 1).cumsum()

    exe_ans_score = float(answer.score.sum()/len(answer))
    step_scores = float(answer.step_score.sum()/len(answer))
    retrieval_score = float(answer[answer.retrieved_value].score.sum()/len(answer[answer.retrieved_value]))
    exe_ans = answer.score.to_list()[-1]

    return {'exe_ans_score': exe_ans_score, 'retrieval_score': retrieval_score, 'exe_ans': exe_ans, 'step_score': step_scores, 'df': answer}


def evaluate_run_results(df_results:pd.DataFrame)->dict:
    """
    Evaluate the results of this run
    """
    results =  {
            'exe_ans_score': float(df_results.exe_ans_score.mean()),
            'retrieval_score': float(df_results.retrieval_score.mean()),
            'exe_ans': float(df_results.exe_ans.mean()),
            'step_score': float(df_results.step_score.mean()),
            'total_completion_tokens':int(df_results.completion_tokens.sum()),
            'total_prompt_tokens':int(df_results.prompt_tokens.sum()),
            'total_tokens':int(df_results.total_tokens.sum()),
            'average_tokens':float(df_results.total_tokens.mean()),
            }

    return {k: v if isinstance(v, int) else round(v, 2) for k,v in results.items()}


def compare_values(table, pct_precision=1, num_precision=1, verbose=False):
    """
    Compares values in each row of the table and returns 1 if they are the same, 0 otherwise.
    Handles both percentage formats and numeric formats.
    Compare values considering floating-point precision

    Args:
        table (list of lists): A list of rows, where each row contains two values.

    Returns:
        list of int: A list of scores (1 if values are the same, 0 otherwise).
    """
    scores = []
    for row in table:
        val1, val2 = row

        if '%' in str(val1) and '%' not in str(val2):
            val1 = float(val1.rstrip('%'))
            val2 = float(f'{val2:.4f}')
            val2 = np.round(float(val2) * 100,1)
            scores.append(1 if abs(float(val1) - float(val2)) < pct_precision else 0)
            if verbose and not scores[-1]:
                print(val1, val2)   
        else:
            val1 = float(val1.strip())
            val2 = np.round(float(val2), 0)
            scores.append(1 if abs(float(val1) - float(val2)) < num_precision else 0)
    return scores


def fix_numerics(val:str, pct:bool = False)->str:
    """
    fix numeric values to have a fixed number of decimal places
    """
    val = val.replace('$ ', '').replace('yes , ', '').strip()
    val = val.replace('increased', '').replace('\\\\n', '').strip()
    val = val.replace('decreased', '').strip()
    
    if '%' in val:
        val = val.replace('%', '').strip()
        pct = True

    print(val)
    try:
        val = float(numerize(val))
        if pct:
            val = f'{val:.1%}'
     except ValueError as e:
        val = 'fail'

    return val


def save_results(df_results:pd.DataFrame, folder:str='./working_results', filename:str=None)->None:
    """
    Save the results to a file
    args:
    df_results: pd.DataFrame: results to save
    folder: str: folder to save the results
    filename: str: name of the file to save the results
    return: None
    """
    ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    if not filename or not isinstance(filename, str):
        filename = f'results_{ts}.csv'
    df_results.to_csv(Path(folder) / filename, index=False)
    return


def remove_extra_spaces(text_in):
    """
    Remove extra spaces from a string.
    Args:
        text_in (str): The input string to process.
    Returns:
        str: The processed string with extra spaces removed.
    """
    return " ".join([s for s in text_in.split if s])


def table_row_to_text(header, row):
    '''
    use templates to convert table row to text
    '''
    res = ""
    
    if header[0]:
        res += (header[0] + " ")

    for head, cell in zip(header[1:], row[1:]):
        res += ("the " + row[0] + " of " + head + " is " + cell.strip() + " ; ")
    
    return remove_extra_spaces(res).strip()
