
def add_response_record(cohort_id, rank_id, response, record_sheet):
    '''
    Add the response of a participant to the record sheet.
    
    Args:
        cohort_id: The cohort number
        rank_id: The participant's rank in the cohort
        response: The participant's response
        record_sheet: The record sheet to update
    '''
    key = (cohort_id, rank_id)
    record_sheet[key]['response'] = response
    return record_sheet

def add_choice_record(cohort_id, rank_id, subject_choice, record_sheet):
    '''
    Add the decision of a participant to the record sheet.
    
    Args:
        cohort_id: The cohort number
        rank_id: The participant's rank in the cohort
        subject_choice: The participant's choice (0 or 1)
        record_sheet: The record sheet to update
    '''
    key = (cohort_id, rank_id)
    record_sheet[key]['subject_choice'] = subject_choice
    return record_sheet
    
def add_payoff_record(cohort_id, rank_id, record_sheet):
    '''
    Calculate and add the payoff of a participant to the record sheet.
    
    Args:
        cohort_id: The cohort number
        rank_id: The participant's rank in the cohort
        record_sheet: The record sheet to update
    '''
    key = (cohort_id, rank_id)

    if record_sheet[key]['subject_choice'] == '0':
        record_sheet[key]['profit'] = 4
    elif record_sheet[key]['subject_choice'] == '1':
        subject_type = record_sheet[key]['subject_type']
        prod_value = record_sheet[key]['product_value']
        wait_time = record_sheet[key]['wait_time']
        payoff = 4 - wait_time + prod_value
        record_sheet[key]['profit'] = payoff

    return record_sheet

def update_wait_time(cohort_id, rank_id, record_sheet):
    '''
    Update the wait time for the next participant in the cohort.
    
    Args:
        cohort_id: The cohort number
        rank_id: The participant's rank in the cohort
        record_sheet: The record sheet to update
    '''
    if rank_id == 4:
        return record_sheet
    else:
        key = (cohort_id, rank_id)
        next_key = (cohort_id, rank_id+1)
    
        wait_time = record_sheet[key]['wait_time']
        if record_sheet[key]['subject_choice'] == '0':
            record_sheet[next_key]['wait_time'] = wait_time
        elif record_sheet[key]['subject_choice'] == '1':
            record_sheet[next_key]['wait_time'] = wait_time+1
        return record_sheet



