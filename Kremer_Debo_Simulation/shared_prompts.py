from experiment_runtime import selection

_FORMATION_BY_CONDITION = selection()["formation_by_condition"]

NAVIGATION_PROMPTS = selection()["navigations"]


def _format_subject_info(subject_type):
    if subject_type == 1:
        return "You are INFORMED."
    return "You are UNINFORMED."


def _format_product_info(subject_type, product_value):
    if subject_type == 1:
        if product_value == 3.5:
            return "You know that this product will be a big hit - every buyer receives $3.50."
        return "You know that this product will be a relative disaster - every buyer receives $0.00."
    return "The product will be a big hit ($3.50) or a relative disaster ($0.00)."


def _format_wait_time_note(wait_time):
    previous_orders = wait_time - 1
    return f"Note: Among the consumer(s) who arrived before you (if any), {previous_orders} order(s) had been placed."


# Q00/Q05/Q50 share a template with condition-specific consumer-type and decision sections.
_CONSUMER_TYPE_TEXT = {
    'Q00': """Consumer type. All consumers are uninformed. Consumers will not learn about the true product value until after the product is received. Before she places an order and before she learns how many other consumers have already ordered the product, all a consumer knows is that the product value is either $0 or $3.50 (each with a 50% probability), i.e., she would expect the product value to be (50% * $0) + (50% * $3.50) = $1.75 (on average).""",
    'Q05': """Consumer Type. For each product, and before any consumer places an order, the computer will also randomly determine the information every consumer has about the product value: With a probability of 5%, a consumer (for example, you) is informed about the product value. An informed consumer will be told about the true product value ($0 or $3.50) before she places an order, and she can simply compare the true product value with the waiting cost based on the delivery time quote. However, with a probability of 95%, a consumer (for example, you) is not informed about the product value. An uninformed consumer will not learn about the true product value until after the product is received. Before she places an order and before she learns how many other consumers have already ordered the product, all an uninformed consumer knows is that the product value is either $0 or $3.50 (each with a 50% probability), i.e., she would expect the product value to be (50% * $0) + (50% * $3.50) = $1.75 (on average).""",
    'Q50': """Consumer Type. For each product, and before any consumer places an order, the computer will also randomly determine the information every consumer has about the product value: With a probability of 50%, a consumer (for example, you) is informed about the product value. An informed consumer will be told about the true product value ($0 or $3.50) before she places an order, and she can simply compare the true product value with the waiting cost based on the delivery time quote. However, with a probability of 50%, a consumer (for example, you) is not informed about the product value. An uninformed consumer will not learn about the true product value until after the product is received. Before she places an order and before she learns how many other consumers have already ordered the product, all an uninformed consumer knows is that the product value is either $0 or $3.50 (each with a 50% probability), i.e., she would expect the product value to be (50% * $0) + (50% * $3.50) = $1.75 (on average).""",
}

_DECISION_SECTION_TEXT = {
    'Q00': """Your Decision (and how you earn)
At the beginning, you receive an endowment of $4 and you learn about your own consumer type. When you arrive at the market and you learn the delivery time, you need to make a decision whether you would want to place an order, or to not place an order.
PLACE NO ORDER. If you decide not to place an order, you simply keep your endowment of $4.
PLACE AN ORDER. If you decide to place an order and the delivery time is X weeks, you will keep your endowment of $4, MINUS the waiting costs, however, PLUS the value of the product (either $0 or $3.50). That is, if the true value of the product is $0 your payoff is $4 - $1X + $0. If the true value of the product is $3.50 your payoff is $4 - $1X + $3.50.""",
    'Q05': """Your Decision (and how you earn)
At the beginning, you receive an endowment of $4 and you learn about your own consumer type (you are either informed and know the value of the product, or you are uninformed). When you arrive at the market and you learn the delivery time, you need to make a decision whether you would want to place an order, or to not place an order.
PLACE NO ORDER. If you decide not to place an order, you simply keep your endowment of $4.
PLACE AN ORDER. If you decide to place an order and the delivery time is X weeks, you will keep your endowment of $4, MINUS the waiting costs, however, PLUS the value of the product (either $0 or $3.50). That is, if the true value of the product is $0 your payoff is $4 - $1X + $0. If the true value of the product is $3.50 your payoff is $4 - $1X + $3.50.""",
    'Q50': """Your Decision (and how you earn)
At the beginning, you receive an endowment of $4 and you learn about your own consumer type (you are either informed and know the value of the product, or you are uninformed). When you arrive at the market and you learn the delivery time, you need to make a decision whether you would want to place an order, or to not place an order.
PLACE NO ORDER. If you decide not to place an order, you simply keep your endowment of $4.
PLACE AN ORDER. If you decide to place an order and the delivery time is X weeks, you will keep your endowment of $4, MINUS the waiting costs, however, PLUS the value of the product (either $0 or $3.50). That is, if the true value of the product is $0 your payoff is $4 - $1X + $0. If the true value of the product is $3.50 your payoff is $4 - $1X + $3.50.""",
}

_INSTRUCTION_TEMPLATE = """You are about to participate in an experiment on decision making. The experiment is designed not to test your knowledge, but to learn about your decision making. 
Task Description
    A firm is facing a market of 4 potential consumers (you and 3 other people in this room). The firm brings a product to the market, which we label with a letter (e.g., 'A-Pod', 'B-Pod', etc.). All potential consumers arrive to the market and have the opportunity to place an order or not and, then, production and delivery of all orders starts. The 4 potential consumers arrive in a random order and at random times at the market. When a potential consumer (for example, you) arrives, the firm quotes a time (in weeks) at which the product will be delivered. Based on this information, the consumer must decide whether to place an order or not. Once all consumer orders are collected, the firm starts production. The firm can produce exactly one unit of the product per week and delivers the products to the consumers, in the sequence that the orders arrived. For example, imagine you arrive and the firm quotes you a delivery time of 2 weeks. Then you know that exactly one other consumer placed an order before you; it takes one week for the firm to deliver this consumer's order, plus one week for your order (if you decide to place one). However, you do not know how many potential consumers have arrived before you. It could be that you are the second consumer and the only consumer that arrived before you placed an order. But it could also be that you are the 4th consumer to arrive, and only 1 out of the 3 consumers that arrived before you placed an order (and 2 consumers did not place an order).
Waiting Costs. If you place an order, then every week you have to wait before receiving the order costs you $1. That is, the value of the product is reduced by $1 times the wait time.
True Product Value. Every consumer that orders eventually receives the product (after some delivery time) and gets the same value from owning the product. Sometimes (with 50% probability), the product is a big hit and has high value of $3.50; sometimes (with 50% probability), the product is a disaster and has low value of $0. For each product, before any consumer places an order, the computer will randomly determine the value that every consumer will receive if she decides to place an order.
{consumer_type}
{decision_section}
"""


def _instruction_for_condition(condition):
    return _INSTRUCTION_TEMPLATE.format(
        consumer_type=_CONSUMER_TYPE_TEXT[condition],
        decision_section=_DECISION_SECTION_TEXT[condition],
    ) + _FORMATION_BY_CONDITION[condition]


def _instruction_q00():
    return _instruction_for_condition('Q00')


def _instruction_q05():
    return _instruction_for_condition('Q05')


def _instruction_q50():
    return _instruction_for_condition('Q50')


_INSTRUCTION_PROMPTS = {
    'Q00': _instruction_q00,
    'Q05': _instruction_q05,
    'Q50': _instruction_q50,
}


def get_instruction_prompt(condition):
    """Render the selected condition's instruction and formation."""
    if condition not in _INSTRUCTION_PROMPTS:
        raise ValueError(f'Unknown condition: {condition}. Choose Q00, Q05 or Q50.')
    return _INSTRUCTION_PROMPTS[condition]()


def get_period_prompt(key, record_sheet, navigation_prompt=None):
    """
    Get the period prompt for a participant.

    Args:
        key: Tuple of (cohort_id, rank_id)
        record_sheet: Dictionary containing all experiment records
        navigation_prompt: Optional string for the navigation guidance line.
                          If None, nav_line is empty (validation default).

    Returns:
        str: The formatted prompt
    """
    wait_time = record_sheet[key]['wait_time']
    subject_type = record_sheet[key]['subject_type']
    product_name = record_sheet[key]['product_name']
    product_value = record_sheet[key]['product_value']

    wait_time_note = _format_wait_time_note(wait_time)
    subject_info = _format_subject_info(subject_type)
    product_info = _format_product_info(subject_type, product_value)

    nav_line = navigation_prompt if navigation_prompt is not None else ""

    period_prompt = f'''You have $4.00.
The firm has introduced the new {product_name}!
The firm offers you a wait time of {wait_time} week(s). {wait_time_note}
{subject_info}
{product_info}
{nav_line}
Given your stepwise explanation first and then put the final decision (ORDER/NOT ORDER) in <answer></answer>.
'''
    return period_prompt
